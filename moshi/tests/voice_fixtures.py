# SPDX-License-Identifier: MIT
"""Synthetic audio for the custom-voice tests: no recordings, no personal data.

speech_like() is a harmonic stack with a 4 Hz syllable envelope. It is not speech, but it has the
level and rhythm the voice-sample selection looks for.
"""

import io
import struct
import wave

import numpy as np

RATE = 24000


def silence(seconds: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(int(round(seconds * rate)), dtype=np.float32)


def tone(freq: float, seconds: float, amplitude: float = 0.5, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speech_like(seconds: float, amplitude: float = 0.3, rate: int = RATE, f0: float = 180.0) -> np.ndarray:
    t = np.arange(int(round(seconds * rate))) / rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    stack = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    return (amplitude * envelope * stack).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def pcm_wav_bytes(samples: np.ndarray, rate: int = RATE, channels: int = 1, width: int = 2) -> bytes:
    """Integer PCM WAV written by the stdlib wave module (8, 16, 24 or 32 bit)."""
    values = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    if channels > 1:
        values = np.repeat(values[:, None], channels, axis=1).reshape(-1)
    scale = {1: 127.0, 2: 32767.0, 3: 8388607.0, 4: 2147483647.0}[width]
    ints = np.rint(values * scale).astype(np.int64)
    if width == 1:
        raw = (ints + 128).astype(np.uint8).tobytes()
    elif width == 2:
        raw = ints.astype("<i2").tobytes()
    elif width == 3:
        raw = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in ints)
    else:
        raw = ints.astype("<i4").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(raw)
    return buffer.getvalue()


def riff(fmt_body: bytes, data: bytes, *, data_size: int | None = None, extra: bytes = b"") -> bytes:
    """A RIFF/WAVE file from a raw fmt body and raw data bytes (for float and extensible headers)."""
    fmt = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
    if len(fmt_body) % 2:
        fmt += b"\x00"
    size = len(data) if data_size is None else data_size
    body = b"WAVE" + extra + fmt + b"data" + struct.pack("<I", size) + data
    if len(data) % 2:
        body += b"\x00"
    return b"RIFF" + struct.pack("<I", len(body)) + body


def float_wav_bytes(samples: np.ndarray, rate: int = RATE, channels: int = 1, width: int = 4) -> bytes:
    dtype = "<f4" if width == 4 else "<f8"
    frames = np.repeat(np.asarray(samples, dtype=np.float64)[:, None], channels, axis=1).reshape(-1)
    fmt = struct.pack("<HHIIHH", 3, channels, rate, rate * channels * width, channels * width, width * 8)
    return riff(fmt, frames.astype(dtype).tobytes())


def extensible_wav_bytes(samples: np.ndarray, *, float_format: bool, rate: int = RATE) -> bytes:
    """WAVE_FORMAT_EXTENSIBLE, mono: 32-bit float or 16-bit PCM. The sub-format GUID says which."""
    bits = 32 if float_format else 16
    width = bits // 8
    subformat = 3 if float_format else 1
    guid = struct.pack("<H", subformat) + bytes.fromhex("0000001000800000aa00389b71")
    fmt = struct.pack("<HHIIHH", 0xFFFE, 1, rate, rate * width, width, bits)
    fmt += struct.pack("<HHI", 22, bits, 4) + guid
    if float_format:
        data = np.asarray(samples, dtype="<f4").tobytes()
    else:
        data = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    return riff(fmt, data)
