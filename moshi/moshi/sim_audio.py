# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# Voice samples for custom voices: WAV decoding, resampling to 24 kHz, and the choice of the
# 4.0 s window that becomes a voice prompt. Torch-free (numpy and the standard library only).
# Nothing here logs audio: callers get arrays and decide what to write.
#
# Why a small RIFF reader and not only the wave module: wave refuses IEEE float and
# WAVE_FORMAT_EXTENSIBLE files, and both are common (DAWs, Drama 3 output, Windows recorders).
# The tests cross-check integer PCM against wave.

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .sim_protocol import SAMPLE_RATE

VOICE_SECONDS = 4.0
VOICE_SAMPLES = int(VOICE_SECONDS * SAMPLE_RATE)  # 96 000 samples: 50 frames of 80 ms
MIN_SPEECH_SECONDS = 2.0

FRAME = 480  # 20 ms analysis frame
HOP = 240  # 10 ms between frame starts
FADE = 480  # 20 ms raised-cosine fades at both ends of the window
SNAP_FRAMES = 10  # a window edge may move up to 100 ms to reach a quiet point
ABS_SPEECH_DB = -45.0  # frames quieter than this are never speech
TARGET_RMS_DBFS = -20.0
PEAK_LIMIT = 0.89  # about -1 dBFS, so the 16-bit sample cannot clip

MAX_CHANNELS = 8
MIN_RATE = 4000
MAX_RATE = 384000

_ZERO_CROSSINGS = 16  # half-length of the resampling filter, in zero crossings of its sinc
_KAISER_BETA = 8.6  # about 90 dB stop-band attenuation
_CHUNK = 8192  # output samples per block in the resampler (bounds the temporaries)

_FORMAT_PCM = 1
_FORMAT_FLOAT = 3
_FORMAT_EXTENSIBLE = 0xFFFE


class AudioError(ValueError):
    """The upload cannot be used as a voice sample. The message holds no audio and no request data."""


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV file to (mono float32 samples, sample rate).

    Accepts integer PCM (8, 16, 24, 32 bit), IEEE float (32, 64 bit), WAVE_FORMAT_EXTENSIBLE with
    either sub-format, any channel count (averaged to mono) and any rate in 4-384 kHz. A data chunk
    that runs past the end of the file (or has a streaming size) is read as far as it goes.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("the upload is not a RIFF/WAVE file")
    fmt: tuple[int, int, int, int] | None = None
    pcm: bytes | None = None
    pos = 12
    while pos + 8 <= len(data) and (fmt is None or pcm is None):
        chunk_id = data[pos:pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        start = pos + 8
        if chunk_id == b"fmt ":
            if size < 16 or start + size > len(data):
                raise AudioError("the WAV fmt chunk is damaged")
            fmt = _parse_fmt(data[start:start + size])
        elif chunk_id == b"data":
            pcm = data[start:min(len(data), start + size)]
        pos = start + size + (size & 1)  # chunks are padded to an even size
    if fmt is None or pcm is None:
        raise AudioError("the WAV file has no fmt or data chunk")
    encoding, channels, rate, bits = fmt
    return _samples(encoding, channels, bits, pcm), rate


def _parse_fmt(body: bytes) -> tuple[int, int, int, int]:
    tag, channels, rate, _byte_rate, _align, bits = struct.unpack_from("<HHIIHH", body, 0)
    if tag == _FORMAT_EXTENSIBLE:
        if len(body) < 26:
            raise AudioError("the WAV extensible header is damaged")
        (tag,) = struct.unpack_from("<H", body, 24)  # first two bytes of the SubFormat GUID
    if not 1 <= channels <= MAX_CHANNELS:
        raise AudioError("the WAV channel count is not supported")
    if not MIN_RATE <= rate <= MAX_RATE:
        raise AudioError("the WAV sample rate is not supported")
    if (tag == _FORMAT_PCM and bits in (8, 16, 24, 32)) or (tag == _FORMAT_FLOAT and bits in (32, 64)):
        return tag, channels, rate, bits
    raise AudioError("unsupported WAV encoding: use 8-32 bit PCM or 32/64 bit float")


def _samples(encoding: int, channels: int, bits: int, pcm: bytes) -> np.ndarray:
    width = bits // 8
    usable = len(pcm) - len(pcm) % (width * channels)
    if usable == 0:
        raise AudioError("the WAV file has no audio samples")
    count = usable // width
    if encoding == _FORMAT_FLOAT:
        values = np.frombuffer(pcm, dtype="<f4" if width == 4 else "<f8", count=count).astype(np.float64)
    elif width == 1:
        values = (np.frombuffer(pcm, dtype=np.uint8, count=count).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(pcm, dtype="<i2", count=count).astype(np.float64) / 32768.0
    elif width == 3:
        raw = np.frombuffer(pcm, dtype=np.uint8, count=usable).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        value = np.where(value >= 1 << 23, value - (1 << 24), value)
        values = value.astype(np.float64) / 8388608.0
    else:
        values = np.frombuffer(pcm, dtype="<i4", count=count).astype(np.float64) / 2147483648.0
    frames = values.reshape(-1, channels)
    mono = frames.mean(axis=1)
    return np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def resample(samples: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited polyphase resampling: windowed sinc (Kaiser), unity gain, no delay.

    Output sample m sits at input position m * rate / target. The filter is cut at the lower of the
    two Nyquist frequencies, so a 15 kHz tone recorded at 44.1 kHz does not alias into 24 kHz audio.
    """
    x = np.asarray(samples, dtype=np.float64)
    if rate == target or x.size == 0:
        return x.astype(np.float32)
    divisor = math.gcd(rate, target)
    up, down = target // divisor, rate // divisor
    peak = max(up, down)
    half = _ZERO_CROSSINGS * peak  # prototype half-length in upsampled samples
    prototype = (up / peak) * np.sinc(np.arange(-half, half + 1) / peak) * np.kaiser(2 * half + 1, _KAISER_BETA)
    reach = -(-half // up)  # input samples on each side that can touch one output sample
    width = 2 * reach + 1
    # Filter taps for each output phase (position modulo up) and each input neighbour.
    offsets = np.arange(up)[:, None] + (reach - np.arange(width))[None, :] * up
    bank = np.where(np.abs(offsets) <= half, prototype[np.clip(offsets + half, 0, 2 * half)], 0.0)
    count = int(round(x.size * up / down))
    position = np.arange(count, dtype=np.int64) * down
    centre = position // up
    phase = position % up
    padded = np.concatenate((np.zeros(reach), x, np.zeros(width + reach)))
    windows = sliding_window_view(padded, width)
    out = np.empty(count, dtype=np.float64)
    for start in range(0, count, _CHUNK):
        stop = min(count, start + _CHUNK)
        out[start:stop] = np.einsum("ij,ij->i", windows[centre[start:stop]], bank[phase[start:stop]])
    return out.astype(np.float32)


def frame_energies_db(x: np.ndarray) -> np.ndarray:
    """Mean-square level of each 20 ms frame (frames start every 10 ms), in dB. Silence is -120 dB."""
    starts = np.arange(0, x.size, HOP)
    ends = np.minimum(starts + FRAME, x.size)
    csum = np.concatenate(([0.0], np.cumsum(np.square(x, dtype=np.float64))))
    energy = (csum[ends] - csum[starts]) / (ends - starts)
    return 10.0 * np.log10(np.maximum(energy, 1e-12))


def speech_frames(energy_db: np.ndarray) -> np.ndarray:
    """Boolean mask of speech frames.

    The threshold sits 40 % of the way from the quiet floor (10th percentile) to the speech level
    (90th percentile), and never below ABS_SPEECH_DB. Continuous speech and silence-with-speech
    both come out right, and a silent file has no speech at all.
    """
    floor = float(np.percentile(energy_db, 10))
    top = float(np.percentile(energy_db, 90))
    threshold = max(floor + 0.4 * (top - floor), ABS_SPEECH_DB)
    return energy_db > threshold


def window_start(n_samples: int, energy_db: np.ndarray, speech: np.ndarray) -> int:
    """First sample of the 4.0 s window.

    Inside the speech region, the window with the highest median frame level wins (silence drags
    the median down, so dense, loud speech wins). Its edges are then nudged by up to 100 ms to the
    quietest point at each end, keeping the length exact. If the speech is shorter than the window,
    the window is centred on it and silence fills the rest.
    """
    speech_idx = np.flatnonzero(speech)
    first, last = int(speech_idx[0]), int(speech_idx[-1])
    window_frames = VOICE_SAMPLES // HOP  # 400 frame starts = 4.0 s
    if last - first + 1 >= window_frames:
        scores = np.median(sliding_window_view(energy_db[first:last + 1], window_frames), axis=1)
        best = first + int(np.argmax(scores))
        # Frame powers; the extra 0 stands for silence past the end of the upload.
        power = np.append(10.0 ** (energy_db / 10.0), 0.0)
        low = max(0, best - SNAP_FRAMES)
        high = min(energy_db.size - window_frames, best + SNAP_FRAMES)
        candidates = np.arange(low, high + 1)
        cost = power[candidates] + power[candidates + window_frames]
        return int(candidates[int(np.argmin(cost))]) * HOP
    centre = ((first + last + 1) * HOP) // 2
    return max(0, min(centre - VOICE_SAMPLES // 2, n_samples - VOICE_SAMPLES))


def _fade(window: np.ndarray) -> None:
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(FADE) / FADE)
    window[:FADE] *= ramp
    window[-FADE:] *= ramp[::-1]


@dataclass(frozen=True)
class VoiceSample:
    pcm: np.ndarray  # int16, VOICE_SAMPLES long, 24 kHz mono
    speech_seconds: float  # speech found in the whole upload
    upload_seconds: float  # the whole upload at 24 kHz
    start_seconds: float  # where the window starts in the upload


def prepare_voice_sample(data: bytes) -> VoiceSample:
    """WAV bytes to the 4.0 s, 24 kHz mono, 16-bit sample a voice prompt is built from.

    Raises AudioError when the upload is not a WAV file or has less than MIN_SPEECH_SECONDS of speech.
    """
    samples, rate = decode_wav(data)
    x = resample(samples, rate).astype(np.float64)
    if x.size == 0:
        raise AudioError("the WAV file has no audio samples")
    x -= x.mean()  # remove DC offset
    energy = frame_energies_db(x)
    speech = speech_frames(energy)
    speech_seconds = int(speech.sum()) * HOP / SAMPLE_RATE
    if speech_seconds < MIN_SPEECH_SECONDS:
        raise AudioError(f"the sample needs at least {MIN_SPEECH_SECONDS:g} s of speech "
                         f"(found {speech_seconds:.1f} s)")
    start = window_start(x.size, energy, speech)
    window = np.zeros(VOICE_SAMPLES, dtype=np.float64)
    taken = x[start:start + VOICE_SAMPLES]
    window[:taken.size] = taken
    _fade(window)
    rms = float(np.sqrt(np.mean(np.square(window))))
    peak = float(np.max(np.abs(window)))
    if rms <= 1e-6 or peak <= 0.0:
        raise AudioError("the sample window is silent")
    gain = min(10.0 ** (TARGET_RMS_DBFS / 20.0) / rms, PEAK_LIMIT / peak)
    pcm = np.clip(np.rint(window * gain * 32767.0), -32768, 32767).astype("<i2")
    return VoiceSample(pcm=pcm, speech_seconds=speech_seconds,
                       upload_seconds=x.size / SAMPLE_RATE, start_seconds=start / SAMPLE_RATE)


def write_wav16(path: str, pcm: np.ndarray) -> None:
    """24 kHz mono 16-bit WAV, the format of every custom voice sample."""
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
