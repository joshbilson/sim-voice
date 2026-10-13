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
| `moshi/moshi/sim_audio.py` (new) | Voice samples: WAV decoding, 24 kHz resampling, the 4.0 s window. Torch-free (numpy, stdlib). |
| `moshi/moshi/sim_voices.py` (new) | Custom voice ids, path rules, the private store (staged builds, atomic install, delete). Torch-free. |
| `moshi/moshi/server.py` | `ServerState` is the engine (`voices`, `run_system_prompts`, `build_voice_prompt`, `reset`, `step`, `new_codec`). Upstream's serve loop moved into `sim_session.py`. `codec=opus` uses sphn as before. |
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

## Low-peak start: converted w8a16 checkpoint

Starting from the bf16 checkpoint and quantizing on the GPU needs the whole bf16 LM and its fp8
copy at once: about 24 GiB of extra memory on a 16.7 GB model. Convert once, then start from the
converted file:

```sh
OMP_NUM_THREADS=2 nice -n 10 python3 scripts/sim_quantize_checkpoint.py \
  --src /path/to/model.safetensors --dst /path/to/w8a16/model.safetensors
python3 scripts/sim_verify_w8a16.py --source /path/to/model.safetensors \
  --converted /path/to/w8a16/model.safetensors --all --gpu
python3 -m moshi.server --fast --host 127.0.0.1 --port 8998 --static none \
  --w8a16-checkpoint /path/to/w8a16/model.safetensors \
  --mimi-weight /path/to/mimi.safetensors --tokenizer /path/to/tokenizer_spm_32k_3.model \
  --voice-prompt-dir /path/to/voices --auth-token-file /path/to/token --default-voice NATF2.pt \
  --default-text-prompt-file /path/to/default-prompt.txt
```

- The converter quantizes on the CPU, one tensor at a time. The 16.7 GB LM converts in about 25 s;
  its own memory is small (the source is memory-mapped and read once, so resident pages are
  reclaimable). Output is written to `<dst>.partial` and renamed when complete.
- The file holds fp8e4m3 weights and fp32 per-channel scales for the quantized layers, under the
  names the modules use, and bf16 for everything else. Metadata: conversion version, source sha256,
  min-features gate, and the LM configuration. The server refuses another version or configuration.
- The server reads each tensor once with `pread` and copies it to the GPU, then drops its page-cache
  range. Nothing is memory-mapped, so start-up peaks at the final footprint, not at the file size.
- `--w8a16-checkpoint` implies `--w8a16` and replaces `--moshi-weight`. It excludes `--fp8` and
  `--cpu-offload`. Without the flag, the online path is unchanged.
- Numerics: quantization divides by a tensor, not a Python scalar, because CUDA rewrites a scalar
  division as a multiply by the reciprocal. CPU and GPU therefore store identical bytes.
  `sim_verify_w8a16.py --all --gpu` checks every quantized tensor (fp8 bytes and fp32 scale, CPU and
  GPU) and every copied bf16 tensor against the source, and exits non-zero on any difference.

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
| `--w8a16-checkpoint PATH` | none | LM already quantized by `scripts/sim_quantize_checkpoint.py`. Implies `--w8a16`; replaces `--moshi-weight`. See the low-peak section. |
| `--ssl DIR` | none | Load `cert.pem` and `key.pem` from `DIR`. Nothing is generated or downloaded. |
| `--custom-voice-dir DIR` | none | Custom voices (`c-<slug>.pt` and `.wav`). Created `0700` if missing. An existing directory must not be group or world accessible. Off without it. See Custom voices. |

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
| `voice_prompt` | a preset `^[A-Za-z0-9_-]{1,40}\.pt$` present in `--voice-prompt-dir`, or a custom `c-<slug>.pt` present in `--custom-voice-dir`. Default `--default-voice` (a preset) |
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
`{"ok":true,"busy":false,"model":"personaplex-7b-v1","fast":true,"voices":["NATF0.pt",...,"c-warm-sim-k3x9.pt"],"customVoices":["c-warm-sim-k3x9.pt"],"uptimeS":123}`.
`voices` is the presets followed by the custom voices; `customVoices` is only the custom ones (empty when `--custom-voice-dir` is not given).

Error responses (JSON `{"code","message"}`) always carry `Connection: close`. A request rejected before its body is read would otherwise leave a keep-alive connection that stalls the next request on it.

## Custom voices

A custom voice is PersonaPlex's voice prompt made from a 4 s sample of the speaker. The server picks the window, runs the same voice-prompt pass upstream runs for the presets (with embedding capture), and writes two files into `--custom-voice-dir`:

| File | Content |
|---|---|
| `c-<slug>.pt` | embeddings and KV cache. Calls load this with `voice_prompt=c-<slug>.pt` |
| `c-<slug>.wav` | the 4.0 s, 24 kHz mono 16-bit sample it was made from (served back by `GET`) |

The id is `c-<slug>` with slug `^[a-z0-9][a-z0-9-]{1,40}$`. Preset names (`NATF2.pt`) resolve only in `--voice-prompt-dir`, and `c-` names only in `--custom-voice-dir`. Every name is validated by regex before it reaches the file system.

| Request (bearer auth, like `/api/chat`) | Result |
|---|---|
| `PUT /api/voices/c-<slug>`, body = WAV, at most 4 MiB | `201 {"voice":"c-<slug>.pt","seconds":4.0,"frames":50,"ms":…}`. Overwrites. `400 bad_audio` (not a WAV, unsupported encoding, under 2 s of speech). `400 bad_request` (id). `403` (a preset). `409 busy`. `413 too_large`. `404` when custom voices are off. `500 internal` (the build failed; the old voice stays) |
| `GET /api/voices/c-<slug>.wav` | the sample as `audio/wav`. `404` if unknown |
| `DELETE /api/voices/c-<slug>` | `204`, both files removed. `404` if unknown, `403` for a preset, `409 busy` while a call or build runs |
| `GET /api/health` | `voices` = presets + custom, `customVoices` = custom |

Accepted input: integer PCM (8, 16, 24, 32 bit), IEEE float (32, 64 bit), WAVE_FORMAT_EXTENSIBLE, any channel count (averaged to mono), 4 to 384 kHz. The stdlib `wave` module refuses float and extensible files, so `sim_audio.py` reads the RIFF chunks itself; the tests cross-check integer PCM against `wave`. Resampling to 24 kHz uses a Kaiser-windowed polyphase sinc (16 zero crossings, about 90 dB stop band).

The window: speech is found from 20 ms frame levels. The threshold sits 40 % of the way from the quiet floor (10th percentile) to the speech level (90th percentile), and never below -45 dBFS. The 4.0 s window with the highest median frame level inside the speech wins. Its edges move up to 100 ms to the quietest points, with 20 ms raised-cosine fades. The result is RMS -20 dBFS with a -1 dBFS peak limit. Under 4 s of speech, the window is centred on it and silence fills the rest. Under 2 s of speech is refused with `400 bad_audio`.

The build: the upload is decoded off the event loop, then takes the busy gate exactly like a call. A call during a build gets `0x05 busy` (4409); a build during a call gets 409. The build runs in its own task, so a client that disconnects cannot release the model while the GPU is still running. Afterwards the streaming state is reset, every random generator is restored, and the loaded voice is forgotten, so the next call loads its own voice. Files are staged in a `.build-*` directory inside the custom directory and moved into place with `rename` (the `.pt` first). An interrupted build leaves no voice behind, and stale staging directories are removed at start.

Requirements and privacy: `--custom-voice-dir` needs `pyloudnorm` (upstream's loader normalises to -24 LUFS, the level the presets were made at); the server refuses to start without it (`pip install pyloudnorm`). Audio is never logged. The build logs `custom voice built: frames=… ms=…` with no id and no name; the call log names the voice as it does for presets. Samples and prompts are private files (0600 inside a 0700 directory).

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
drops, error codes, opus routing, start-up refusals and the client script end to end. For custom
voices it covers WAV decoding (every supported encoding, malformed files), resampling (a tone's
frequency and level, alias rejection), the speech window on synthetic signals, the store (ids,
paths, permissions, atomic install, delete), the HTTP routes with the busy gate in both directions,
failed builds, and that the next call still runs its system prompts after a build.

With torch installed (CPU is enough), `test_w8a16_checkpoint.py` also runs: quantization bit-identity,
the converted file against the online path (both LM layouts, state and forward outputs), the
original loader patches, and the refusals. Without torch those tests skip. Set `W8A16_SOURCE` and
`W8A16_CONVERTED` to add a sampled real-tensor check.

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
