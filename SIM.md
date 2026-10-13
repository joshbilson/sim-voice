# Sim voice fork

This branch (`sim`) turns PersonaPlex into a server that one personal voice agent can call over
a token-authenticated WebSocket: 16-bit PCM framing, bounded latency, one call at a time. The
upstream web client (`codec=opus`, Ogg Opus) keeps working unchanged.

Model, quantization and loader code is upstream's, unchanged (`lm.py`, `loaders.py`,
`w8a16_quantize.py`, `fp8_quantize.py`).

## What changed from upstream

| Area | Change |
|---|---|
| `moshi/moshi/sim_protocol.py` (new) | Wire constants, request validation, bearer-token helpers, `Pcm16Codec`, frame buffer, latency window. Torch-free. |
| `moshi/moshi/sim_session.py` (new) | HTTP and WebSocket layer: auth, 400/401 before upgrade, busy (4409), idle and session limits, stats, error codes. Torch-free: the engine owns the model and Opus. |
| `moshi/moshi/sim_cli.py` (new) | Flags and start-up refusals (non-loopback host needs a token file, token file permissions, gradio flags refused). Torch-free. |
| `moshi/moshi/server.py` | `ServerState` is the engine (`voices`, `run_system_prompts`, `reset`, `step`, `new_codec`). Upstream's serve loop moved into `sim_session.py`. `codec=opus` uses sphn as before. |
| `moshi/moshi/offline.py` | The HF "download counter" request (`config.json`) is removed. |
| Removed | gradio tunnel (refused with an error), HF download counter, static web UI download (`--static` defaults to `none`), mkcert download and `mkcert -install` (`--ssl` loads an existing `cert.pem`/`key.pem` or refuses), request-line access logs (they carry the text prompt). |
| `scripts/sim_call_test.py` (new) | Drives one call with `codec=pcm16`, streams a WAV in real time, records audio and text, prints a JSON summary. Needs aiohttp and numpy. |
| `moshi/tests/` (new) | stdlib `unittest` suite with a fake engine (no torch, no GPU). |

## Running

Local weights and voices, no network:

```sh
HF_HUB_OFFLINE=1 python -m moshi.server --fast \
  --host 127.0.0.1 --port 8998 --static none \
  --moshi-weight /path/to/model.safetensors \
  --mimi-weight /path/to/mimi.safetensors \
  --tokenizer /path/to/tokenizer_spm_32k_3.model \
  --voice-prompt-dir /path/to/voices \
  --auth-token-file /path/to/token --default-voice NATF2.pt \
  --default-text-prompt-file /path/to/default-prompt.txt
```

A non-loopback `--host` refuses to start without `--auth-token-file`. The token file must be
chmod 600 (no group or world access) and at least 16 printable characters, with no spaces.
Generate one with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.

Smoke test without weights (fake engine, echoes audio):

```sh
python3 moshi/tests/run_fake_server.py --port 8998 --auth-token-file /path/to/token
python3 scripts/sim_call_test.py --url http://127.0.0.1:8998 --token-file /path/to/token \
  --wav call.wav --voice NATF2.pt --out-wav out.wav --out-text out.txt
```

## Flags added by the fork

| Flag | Default | Meaning |
|---|---|---|
| `--auth-token-file PATH` | none | Bearer token for every `/api/*` request. Required when `--host` is not loopback. |
| `--default-voice NAME` | `NATF2.pt` | Voice used when a call does not name one. Must exist in the voice directory. |
| `--default-text-prompt-file PATH` | built-in sentence | Default persona text, at most 1200 characters. |
| `--max-backlog-frames N` | `6` | Input frames (80 ms each) allowed to wait. Older frames are dropped and counted. |
| `--idle-timeout-s S` | `15` | End the call (`timeout`, 4408) after this long without client audio. |
| `--max-session-seconds S` | `1800` | End the call (`timeout`, 4408) after this long. |
| `--stats-interval-s S` | `2` | Seconds between stats messages (`0x06`). |
| `--static DIR\|none` | `none` | Optional web UI directory. Never downloaded. |
| `--ssl DIR` | none | Load `cert.pem` and `key.pem` from `DIR`. Nothing is generated or downloaded. |

Removed: `--gradio-tunnel` and `--gradio-tunnel-token` now fail with an error.
Unchanged from upstream: `--fast`, `--w8a16`, `--fp8`, `--skip-other-mimi`, `--mimi-fp16`,
`--pinned-io`, `--dep-q-exit`, `--cpu-offload`, `--device`, the weight and tokenizer paths.

`--fast` (upstream GB10 preset) requires CUDA and Triton and refuses to start without them.

## Protocol

WebSocket `GET /api/chat`, binary messages, first byte is the kind. Auth is
`Authorization: Bearer <token>`, compared in constant time; a wrong or missing token gets HTTP 401
before the upgrade.

Query parameters (invalid values get HTTP 400 before the upgrade; an empty value means "absent"):

| Param | Rule |
|---|---|
| `codec` | `pcm16` (Sim) or `opus` (default; upstream web client) |
| `voice_prompt` | `^[A-Za-z0-9_-]{1,40}\.pt$`, present in the voice directory. Default `--default-voice` |
| `text_prompt` | at most 1200 characters. Default `--default-text-prompt-file` |
| `seed` | `-1` (unseeded) or an integer in `0..4294967295` |
| `session` | `^[A-Za-z0-9-]{1,64}$`. Used in logs only. Generated when absent |

Kinds:

| Byte | Direction | Payload |
|---|---|---|
| `0x04` | server to client | JSON, first message after the upgrade: `{"type":"accepted","session","sampleRate":24000,"frameMs":80,"codec","voice","model":"personaplex-7b-v1","fast"}` |
| `0x00` | server to client | Handshake: persona prompts processed, the call is live. Empty payload |
| `0x11` | both | `pcm16`: signed 16-bit little-endian mono, 24 kHz. Client: any length. Server: one 80 ms frame (3840 bytes) per message |
| `0x01` | both | `opus` only: Ogg Opus, as upstream |
| `0x02` | server to client | UTF-8 text piece, the model's own words |
| `0x06` | server to client | Stats about every 2 s: `{"type":"stats","msPerFrameP50","backlogFrames","droppedFrames"}`. `msPerFrameP50` is `null` until the first frame is processed |
| `0x05` | server to client | JSON `{"code","message"}`, then close. Codes: `busy`, `bad_request`, `timeout`, `internal` |

Client audio sent before `0x00` is discarded. Close codes: 1000 normal, 4400 bad request, 4408
timeout, 4409 busy, 4500 internal.

Behaviour:

- One call at a time. A second connection gets `0x05 busy` and close 4409 at once. There is no queue.
- Input is bounded: when more than `--max-backlog-frames` frames wait, the oldest is dropped and counted.
- No client audio for `--idle-timeout-s` after `0x00` ends the call with `timeout` (4408).
- After `--max-session-seconds` the call ends with `timeout` (4408) and the message `max session length reached`.
- An undecodable audio payload ends the call with `bad_request` (4400).
- Audio of the other codec's kind is ignored (`0x11` on an `opus` call, `0x01` on a `pcm16` call).
- Engine failure ends the call with `internal` (4500). The message is generic; the log holds only the exception type and location.
- On `codec=opus` the server also sends `0x04`, `0x06` and `0x05`. The upstream web client tolerates them (its console logs a parse error for `0x04`).

`GET /api/health` (auth required) returns
`{"ok":true,"busy":false,"model":"personaplex-7b-v1","fast":true,"voices":["NATF0.pt",...],"uptimeS":123}`.

## Privacy

- Logs hold session ids, timings, frame counts, close codes and the voice name. They never hold text prompts, transcripts, audio or the token. The test suite checks this by capturing logs during a call.
- Request-line access logs are off (`access_log=None`), because the text prompt travels in the query string.
- With local weight, tokenizer and voice paths, no Hugging Face request is made. `HF_HUB_OFFLINE=1` makes any missed path fail instead of downloading.
- The token is never printed. The client script reads it from a file and prints a summary without it.

## Tests

```sh
python3 -m unittest discover -s moshi/tests -v
```

Needs only `aiohttp` and `numpy` (the pins in `pyproject.toml`). No torch and no GPU. The suite
covers auth, validation, the handshake, PCM16 framing, busy, idle and session timeouts, backlog
drops, error codes, opus routing, start-up refusals and the client script end to end.

## Licences and attribution

- Code: MIT, as in `LICENSE-MIT`. NVIDIA's copyright notice is in the file headers (for example
  `moshi/moshi/server.py`). Kyutai's Moshi code keeps `moshi/LICENSE.moshi`, and audiocraft-derived
  code keeps `moshi/LICENSE.audiocraft`. Nothing in those licences has been changed.
- Model weights: NVIDIA Open Model License (see the README). Weights are not in this repository.
- GB10 real-time work this branch is based on (git branch `jethac/pr-gb10-realtime`): Jetha Chan
  (`jethac`) for `--fast`, the w8a16 GEMV, depformer early exit and the GB10 hints; amarrmb for FP8,
  pinned I/O and mimi compile. Thanks to both.
- The Sim protocol, session layer, CLI, client script and tests are new in this fork. They carry the
  MIT SPDX line.

## Keeping this repository public

Personal configuration never goes in this repository: no tokens, hostnames, tailnet addresses,
user paths, prompts or transcripts. Deployment values live outside the repository and are passed
on the command line or in files such as `/path/to/...` in the examples above.
