# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# Wire protocol, request validation, bearer-token helpers and audio codecs for the Sim voice
# server. Pure Python + numpy: no torch, sphn or sentencepiece, so it is testable without a GPU.
# Contract summary: SIM.md. Nothing here logs or stores user content.

from __future__ import annotations

import collections
import hmac
import ipaddress
import json
import os
import re
import secrets
import stat
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920  # 80 ms at 24 kHz
FRAME_MS = 80
MODEL_NAME = "personaplex-7b-v1"

DEFAULT_VOICE = "NATF2.pt"
DEFAULT_TEXT_PROMPT = "You enjoy having a good conversation."
MAX_TEXT_PROMPT_CHARS = 1200
MIN_TOKEN_CHARS = 16
CODECS = ("opus", "pcm16")

# fullmatch() is used everywhere: '$' would also accept a trailing newline.
VOICE_RE = re.compile(r"[A-Za-z0-9_-]{1,40}\.pt")
SESSION_RE = re.compile(r"[A-Za-z0-9-]{1,64}")
TOKEN_RE = re.compile(r"[\x21-\x7e]{%d,}" % MIN_TOKEN_CHARS)

# Message kinds (first byte of every binary WebSocket message).
KIND_HANDSHAKE = 0x00  # S->C, empty: persona prompts processed, the call is live
KIND_OPUS = 0x01  # both, Ogg Opus (codec=opus only)
KIND_TEXT = 0x02  # S->C, UTF-8 text piece
KIND_ACCEPTED = 0x04  # S->C, JSON, first message after the upgrade
KIND_ERROR = 0x05  # S->C, JSON {"code","message"}, then close
KIND_STATS = 0x06  # S->C, JSON about every 2 s
KIND_PCM16 = 0x11  # both, s16le mono 24 kHz (codec=pcm16)

# Close codes.
CLOSE_NORMAL = 1000
ERROR_CLOSE_CODES = {"bad_request": 4400, "timeout": 4408, "busy": 4409, "internal": 4500}


class BadRequest(ValueError):
    """A request parameter failed validation. The message never echoes user input."""


class TokenFileError(ValueError):
    """The bearer token file is missing, too open, or malformed. Never contains the token."""


def json_message(kind: int, obj: Mapping) -> bytes:
    return bytes([kind]) + json.dumps(obj, separators=(",", ":")).encode("utf-8")


def error_message(code: str, message: str) -> bytes:
    return json_message(KIND_ERROR, {"code": code, "message": message})


def make_session_id() -> str:
    return secrets.token_hex(8)


@dataclass(frozen=True)
class ChatParams:
    codec: str
    voice_prompt: str
    text_prompt: str
    seed: Optional[int]
    session: str


def validate_voice_prompt(name: str, voices: Iterable[str]) -> str:
    if not VOICE_RE.fullmatch(name):
        raise BadRequest("voice_prompt must look like NATF2.pt")
    if name not in set(voices):
        raise BadRequest("voice_prompt is not available on this server")
    return name


def resolve_text_prompt(raw: Optional[str], default: str) -> str:
    if raw is None or raw.strip() == "":
        return default
    if len(raw) > MAX_TEXT_PROMPT_CHARS:
        raise BadRequest(f"text_prompt is longer than {MAX_TEXT_PROMPT_CHARS} characters")
    return raw


def parse_seed(raw: Optional[str]) -> Optional[int]:
    """Absent or -1 means unseeded. Otherwise an integer in [0, 2**32)."""
    if raw is None or raw == "":
        return None
    try:
        value = int(raw, 10)
    except ValueError:
        raise BadRequest("seed must be an integer") from None
    if value == -1:
        return None
    if not 0 <= value < 2**32:
        raise BadRequest("seed is out of range")
    return value


def validate_session(raw: Optional[str]) -> Optional[str]:
    if raw is None or raw == "":
        return None
    if not SESSION_RE.fullmatch(raw):
        raise BadRequest("session must be 1-64 letters, digits or dashes")
    return raw


def parse_chat_query(
    query: Mapping[str, str],
    *,
    voices: Iterable[str],
    default_voice: str = DEFAULT_VOICE,
    default_text_prompt: str = DEFAULT_TEXT_PROMPT,
) -> ChatParams:
    """Validate /api/chat query parameters. Raises BadRequest; the caller answers HTTP 400."""
    codec = query.get("codec") or "opus"
    if codec not in CODECS:
        raise BadRequest("codec must be opus or pcm16")
    voice_raw = query.get("voice_prompt") or default_voice
    voice = validate_voice_prompt(voice_raw, voices)
    text = resolve_text_prompt(query.get("text_prompt"), default_text_prompt)
    seed = parse_seed(query.get("seed"))
    session = validate_session(query.get("session")) or make_session_id()
    return ChatParams(codec=codec, voice_prompt=voice, text_prompt=text, seed=seed, session=session)


def is_loopback_host(host: str) -> bool:
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def bearer_matches(authorization: Optional[str], expected: bytes) -> bool:
    """Constant-time check of an 'Authorization: Bearer <token>' header value. Fails closed."""
    if not expected or not authorization:
        return False
    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(presented.strip().encode("utf-8", "surrogateescape"), expected)


def load_token_file(path: str) -> str:
    """Read the bearer token. Refuses group/world-accessible files and short or unsafe tokens."""
    file_path = Path(path)
    try:
        st = os.stat(file_path)
    except OSError:
        raise TokenFileError("auth token file cannot be read") from None
    if not stat.S_ISREG(st.st_mode):
        raise TokenFileError("auth token path is not a regular file")
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise TokenFileError("auth token file must not be group or world accessible (chmod 600)")
    token = file_path.read_text(encoding="utf-8").strip()
    if not TOKEN_RE.fullmatch(token):
        raise TokenFileError(
            f"auth token must be at least {MIN_TOKEN_CHARS} printable characters with no spaces"
        )
    return token


def read_default_text_prompt(path: Optional[str]) -> str:
    """Contents of --default-text-prompt-file; the built-in default when no file is given."""
    if path is None:
        return DEFAULT_TEXT_PROMPT
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return DEFAULT_TEXT_PROMPT
    if len(text) > MAX_TEXT_PROMPT_CHARS:
        raise ValueError(f"default text prompt is longer than {MAX_TEXT_PROMPT_CHARS} characters")
    return text


class Pcm16Codec:
    """Client <-> server audio as signed 16-bit little-endian mono PCM (codec=pcm16).

    Input messages may have any length; an odd trailing byte is carried to the next message.
    Output is one decoded frame per message, clipped to int16.
    """

    in_kind = KIND_PCM16
    out_kind = KIND_PCM16

    def __init__(self) -> None:
        self._carry = b""

    def decode(self, payload: bytes) -> np.ndarray:
        data = self._carry + payload if self._carry else payload
        count = len(data) // 2
        self._carry = data[2 * count:]
        if count == 0:
            return np.zeros(0, dtype=np.float32)
        samples = np.frombuffer(data, dtype="<i2", count=count)
        return samples.astype(np.float32) * np.float32(1.0 / 32768.0)

    def encode(self, pcm: np.ndarray) -> bytes:
        scaled = np.asarray(pcm, dtype=np.float32) * np.float32(32768.0)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=32767.0, neginf=-32768.0)
        np.rint(scaled, out=scaled)
        np.clip(scaled, -32768.0, 32767.0, out=scaled)
        return scaled.astype("<i2").tobytes()


class FrameBuffer:
    """Input backlog in whole frames. When more than max_frames wait, the oldest is dropped."""

    def __init__(self, frame_samples: int = FRAME_SAMPLES, max_frames: int = 6) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        self.frame_samples = frame_samples
        self.max_frames = max_frames
        self.dropped = 0
        self._partial = np.zeros(0, dtype=np.float32)
        self._frames: collections.deque = collections.deque()

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, pcm: np.ndarray) -> int:
        """Append samples; returns how many frames this push dropped."""
        if pcm.size == 0:
            return 0
        data = np.concatenate((self._partial, pcm.astype(np.float32, copy=False)))
        full = data.size // self.frame_samples
        dropped_now = 0
        for index in range(full):
            start = index * self.frame_samples
            self._frames.append(data[start:start + self.frame_samples].copy())
            while len(self._frames) > self.max_frames:
                self._frames.popleft()
                self.dropped += 1
                dropped_now += 1
        self._partial = data[full * self.frame_samples:].copy()
        return dropped_now

    def pop(self) -> Optional[np.ndarray]:
        return self._frames.popleft() if self._frames else None


class LatencyWindow:
    """Rolling window of per-frame processing times, in milliseconds."""

    def __init__(self, size: int = 100) -> None:
        self._values: collections.deque = collections.deque(maxlen=size)

    def add(self, ms: float) -> None:
        self._values.append(ms)

    def p50(self) -> Optional[float]:
        if not self._values:
            return None
        return round(statistics.median(self._values), 1)
