#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Sim fork (joshbilson/sim-voice). Drives one call against a sim-voice server with codec=pcm16.
"""Drive one Sim voice call and print a JSON summary.

Connects to /api/chat, waits for the handshake (0x00), streams a WAV in real time as 80 ms
frames, then sends silence until --seconds have passed, while recording the server's audio and
text. Needs only aiohttp and numpy.

The token is read from --token-file and is never printed. Text and audio are written only to the
files named by --out-text and --out-wav; the summary holds counts, timings and codes.

Timings in the summary are milliseconds from the start of the connection attempt:
  msToHandshake  WebSocket upgrade complete
  msToAccepted   0x04 accepted message
  msToLive       0x00 handshake (persona prompts processed, the call is live)
  msToFirstAudio first 0x11 audio frame from the server

Exit status: 0 call went live, produced audio and closed with 1000 with no error message;
1 error message from the server, or a close code other than 1000, or no audio;
2 handshake refused (for example HTTP 401 or 400) or bad usage.

Example:
  python3 scripts/sim_call_test.py --url http://127.0.0.1:8998 --token-file ~/.config/sim-voice/token \\
      --wav call.wav --voice NATF2.pt --text-prompt "You are a calm assistant." --out-wav out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
import numpy as np

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920  # 80 ms
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE
SILENCE_FRAME = bytes(FRAME_SAMPLES * 2)

KIND_HANDSHAKE = 0x00
KIND_TEXT = 0x02
KIND_ACCEPTED = 0x04
KIND_ERROR = 0x05
KIND_STATS = 0x06
KIND_AUDIO = 0x11


class UsageError(Exception):
    pass


def read_token(path: str) -> str:
    mode = os.stat(path).st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise UsageError("token file must not be group or world accessible (chmod 600)")
    token = Path(path).read_text(encoding="utf-8").strip()
    if not token:
        raise UsageError("token file is empty")
    return token


def build_headers(items: list[str], token: Optional[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in items:
        name, sep, value = item.partition(":")
        if not sep or not name.strip():
            raise UsageError("--header must look like 'Name: value'")
        headers[name.strip()] = value.strip()
    if token is not None:
        if any(name.lower() == "authorization" for name in headers):
            raise UsageError("use either --token-file or an Authorization --header, not both")
        headers["Authorization"] = f"Bearer {token}"
    return headers


def chat_url(base: str, params: dict[str, str]) -> str:
    parts = urlsplit(base)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parts.scheme)
    if scheme is None or not parts.netloc:
        raise UsageError("--url must start with http(s):// or ws(s)://")
    path = parts.path.rstrip("/")
    if not path.endswith("/api/chat"):
        path += "/api/chat"
    return urlunsplit((scheme, parts.netloc, path, urlencode(params), ""))


def read_wav_24k_mono(path: str) -> np.ndarray:
    """float32 mono samples at 24 kHz. Downmixes and resamples (linear) when needed."""
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if width == 1:
        samples = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        value = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        value = np.where(value & 0x800000, value - 0x1000000, value)
        samples = value.astype(np.float32) / 8388608.0
    elif width == 4:
        samples = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise UsageError(f"unsupported WAV sample width {width} bytes")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE and samples.size:
        out_len = max(1, round(samples.size * SAMPLE_RATE / rate))
        positions = np.arange(out_len) * (rate / SAMPLE_RATE)
        samples = np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
    return samples


def to_frames(samples: np.ndarray) -> list[bytes]:
    """80 ms frames as s16le bytes; the last frame is padded with silence."""
    frames = []
    for start in range(0, samples.size, FRAME_SAMPLES):
        chunk = samples[start:start + FRAME_SAMPLES]
        if chunk.size < FRAME_SAMPLES:
            chunk = np.concatenate((chunk, np.zeros(FRAME_SAMPLES - chunk.size, np.float32)))
        pcm = np.clip(np.rint(chunk * 32768.0), -32768, 32767).astype("<i2")
        frames.append(pcm.tobytes())
    return frames


def write_wav(path: str, pcm_bytes: bytes) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm_bytes)


class Recorder:
    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.accepted: Optional[dict] = None
        self.ms_accepted: Optional[float] = None
        self.ms_live: Optional[float] = None
        self.ms_first_audio: Optional[float] = None
        self.frames_out = 0
        self.audio: list[bytes] = []
        self.text: list[str] = []
        self.last_stats: Optional[dict] = None
        self.error: Optional[dict] = None
        self.live = asyncio.Event()
        self.finished = asyncio.Event()

    def ms(self) -> float:
        return round((time.monotonic() - self.t0) * 1000.0, 1)


def handle_message(rec: Recorder, kind: int, body: bytes) -> None:
    if kind == KIND_ACCEPTED:
        rec.accepted = json.loads(body)
        rec.ms_accepted = rec.ms()
    elif kind == KIND_HANDSHAKE:
        rec.ms_live = rec.ms()
        rec.live.set()
    elif kind == KIND_AUDIO:
        if rec.frames_out == 0:
            rec.ms_first_audio = rec.ms()
        rec.frames_out += 1
        rec.audio.append(body)
    elif kind == KIND_TEXT:
        rec.text.append(body.decode("utf-8", "replace"))
    elif kind == KIND_STATS:
        rec.last_stats = json.loads(body)
    elif kind == KIND_ERROR:
        rec.error = json.loads(body)


async def read_messages(ws: aiohttp.ClientWebSocketResponse, rec: Recorder) -> None:
    try:
        async for msg in ws:
            if msg.type is not aiohttp.WSMsgType.BINARY or not msg.data:
                continue
            try:
                handle_message(rec, msg.data[0], msg.data[1:])
            except ValueError:
                if msg.data[0] == KIND_ERROR:
                    rec.error = {"code": "unparsed", "message": ""}
    finally:
        rec.finished.set()


async def stream_audio(ws: aiohttp.ClientWebSocketResponse, rec: Recorder, frames: list[bytes],
                       total_frames: int, live_timeout: float) -> int:
    live_wait = asyncio.create_task(rec.live.wait())
    gone_wait = asyncio.create_task(rec.finished.wait())
    await asyncio.wait({live_wait, gone_wait}, timeout=live_timeout, return_when=asyncio.FIRST_COMPLETED)
    for task in (live_wait, gone_wait):
        task.cancel()
    await asyncio.gather(live_wait, gone_wait, return_exceptions=True)
    if not rec.live.is_set():
        return 0

    sent = 0
    start = time.monotonic()
    for index in range(total_frames):
        delay = start + index * FRAME_S - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        if rec.finished.is_set():
            break
        payload = frames[index] if index < len(frames) else SILENCE_FRAME
        try:
            await ws.send_bytes(bytes([KIND_AUDIO]) + payload)
        except (ConnectionError, RuntimeError, aiohttp.ClientError):
            break  # the server closed the call
        sent += 1
    return sent


async def run_call(args: argparse.Namespace, headers: dict[str, str], frames: list[bytes],
                   total_frames: int) -> tuple[dict, int, Optional[Recorder]]:
    session_id = args.session or str(uuid.uuid4())
    params = {"codec": "pcm16", "session": session_id}
    if args.voice:
        params["voice_prompt"] = args.voice
    if args.text_prompt is not None:
        params["text_prompt"] = args.text_prompt
    if args.seed is not None:
        params["seed"] = str(args.seed)
    url = chat_url(args.url, params)

    t0 = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        try:
            ws = await http.ws_connect(url, headers=headers)
        except aiohttp.WSServerHandshakeError as exc:
            return {"ok": False, "session": session_id, "error": "handshake_refused",
                    "status": exc.status, "reason": exc.message}, 2, None
        except aiohttp.ClientError as exc:
            return {"ok": False, "session": session_id, "error": "connect_failed",
                    "detail": type(exc).__name__}, 2, None

        rec = Recorder(t0)
        ms_handshake = rec.ms()
        reader = asyncio.create_task(read_messages(ws, rec))
        sent = await stream_audio(ws, rec, frames, total_frames, args.live_timeout)
        if not rec.finished.is_set() and sent:
            await asyncio.sleep(args.tail_seconds)
        if not ws.closed:
            await ws.close()
        try:
            await asyncio.wait_for(reader, timeout=5.0)
        except Exception:  # timeout or a read error: the close code below still tells the story
            reader.cancel()
        close_code = ws.close_code

    summary = {
        "ok": False,
        "session": session_id,
        "msToHandshake": ms_handshake,
        "msToAccepted": rec.ms_accepted,
        "msToLive": rec.ms_live,
        "msToFirstAudio": rec.ms_first_audio,
        "framesIn": sent,
        "framesOut": rec.frames_out,
        "textPieces": len(rec.text),
        "textChars": sum(len(piece) for piece in rec.text),
        "accepted": rec.accepted,
        "lastStats": rec.last_stats,
        "errorCode": rec.error.get("code") if rec.error else None,
        "errorMessage": rec.error.get("message") if rec.error else None,
        "closeCode": close_code,
    }
    summary["ok"] = bool(
        rec.accepted is not None and rec.ms_live is not None and rec.error is None
        and close_code == 1000 and rec.frames_out > 0
    )
    return summary, (0 if summary["ok"] else 1), rec


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive one Sim voice call (codec=pcm16) and print a JSON summary.")
    parser.add_argument("--url", required=True, help="Server base URL, e.g. http://127.0.0.1:8998 (/api/chat is added).")
    parser.add_argument("--token-file", help="Bearer token file (chmod 600). Read, never printed.")
    parser.add_argument("--header", action="append", default=[], metavar="K:V",
                        help="Extra request header, e.g. 'X-Example: 1'. Repeatable.")
    parser.add_argument("--wav", help="WAV to stream (24 kHz mono 16-bit preferred; others are converted). Omit for silence.")
    parser.add_argument("--voice", help="voice_prompt, e.g. NATF2.pt")
    parser.add_argument("--text-prompt", help="text_prompt for this call (<= 1200 characters)")
    parser.add_argument("--seed", type=int, help="seed (-1 means unseeded)")
    parser.add_argument("--session", help="session id (letters, digits, dashes). Default: random UUID.")
    parser.add_argument("--seconds", type=float,
                        help="Seconds of audio to stream after the handshake. Default: WAV length + 2, or 10 without a WAV.")
    parser.add_argument("--tail-seconds", type=float, default=1.0,
                        help="Seconds to keep listening after the last frame is sent.")
    parser.add_argument("--live-timeout", type=float, default=60.0,
                        help="Seconds to wait for the handshake before giving up.")
    parser.add_argument("--out-wav", help="Write the server's audio here (24 kHz mono 16-bit WAV).")
    parser.add_argument("--out-text", help="Write the server's text pieces here (UTF-8).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        token = read_token(args.token_file) if args.token_file else None
        headers = build_headers(args.header, token)
        samples = read_wav_24k_mono(args.wav) if args.wav else np.zeros(0, np.float32)
    except (UsageError, OSError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    frames = to_frames(samples)
    if args.seconds is not None:
        seconds = args.seconds
    else:
        seconds = (samples.size / SAMPLE_RATE + 2.0) if args.wav else 10.0
    total_frames = max(1, int(np.ceil(seconds / FRAME_S)))

    summary, code, rec = asyncio.run(run_call(args, headers, frames, total_frames))
    if rec is not None:
        if args.out_wav:
            write_wav(args.out_wav, b"".join(rec.audio))
        if args.out_text:
            Path(args.out_text).write_text("".join(rec.text), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
