# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# Torch-free safetensors reader and streaming writer for the offline w8a16 checkpoint
# (w8a16_checkpoint.py). The writer takes raw bytes one tensor at a time, so a converter never
# holds the whole model in memory. sim_cli uses the reader to refuse a bad file before any model
# loading starts. Format: https://github.com/huggingface/safetensors (8-byte little-endian header
# length, JSON header, contiguous data).

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from typing import Iterable, Mapping

# Bump when the layout of the converted file changes. The loader refuses any other value.
CONVERSION_VERSION = "1"
METADATA_PREFIX = "moshi_w8a16_"
MAX_HEADER_BYTES = 100 * 1024 * 1024

ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class CheckpointFormatError(ValueError):
    """The file is not a usable safetensors checkpoint, or not one this code can load."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple
    begin: int  # offsets relative to the start of the data section
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class SafetensorsHeader:
    metadata: dict
    tensors: dict  # name -> TensorInfo, in file order
    data_start: int
    file_size: int


def _numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def read_header(path: str) -> SafetensorsHeader:
    """Parse and validate a safetensors header without reading tensor data."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise CheckpointFormatError("file is shorter than a safetensors header")
            (header_len,) = struct.unpack("<Q", raw)
            if header_len <= 2 or header_len > MAX_HEADER_BYTES or 8 + header_len > size:
                raise CheckpointFormatError("implausible safetensors header length")
            header = json.loads(handle.read(header_len).decode("utf-8"))
    except OSError as exc:
        raise CheckpointFormatError(f"cannot read checkpoint ({type(exc).__name__})") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise CheckpointFormatError("safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise CheckpointFormatError("safetensors header is not an object")

    metadata = header.pop("__metadata__", {}) or {}
    if not isinstance(metadata, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
        raise CheckpointFormatError("safetensors metadata must map strings to strings")

    data_start = 8 + header_len
    data_size = size - data_start
    parsed = []
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise CheckpointFormatError(f"tensor entry {name!r} is malformed")
        try:
            dtype = entry["dtype"]
            shape = tuple(int(dim) for dim in entry["shape"])
            begin, end = (int(x) for x in entry["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError(f"tensor entry {name!r} is malformed") from exc
        if dtype not in ITEMSIZE:
            raise CheckpointFormatError(f"tensor {name!r} has unsupported dtype {dtype}")
        if end < begin or end - begin != _numel(shape) * ITEMSIZE[dtype]:
            raise CheckpointFormatError(f"tensor {name!r} size does not match its shape and dtype")
        parsed.append((begin, end, name, TensorInfo(dtype, shape, begin, end)))

    tensors: dict[str, TensorInfo] = {}
    expected_begin = 0
    for begin, end, name, info in sorted(parsed, key=lambda item: (item[0], item[1])):
        if begin != expected_begin:
            raise CheckpointFormatError(f"tensor {name!r} offsets are not contiguous")
        tensors[name] = info
        expected_begin = end
    if expected_begin != data_size:
        raise CheckpointFormatError("tensor data does not fill the file")
    return SafetensorsHeader(metadata, tensors, data_start, size)


def check_w8a16_metadata(metadata: Mapping[str, str]) -> None:
    """Refuse a converted file made by another layout version."""
    version = metadata.get(METADATA_PREFIX + "version")
    if version != CONVERSION_VERSION:
        raise CheckpointFormatError(
            f"not a w8a16 checkpoint for this server (version {version!r}, need {CONVERSION_VERSION!r}); "
            "make it with scripts/sim_quantize_checkpoint.py")


class SafetensorsWriter:
    """Streams a safetensors file. Entries are declared up front (the header holds the offsets)
    and their bytes must then be written in the same order, exactly nbytes each."""

    def __init__(self, path: str, entries: list, metadata: Mapping[str, str]):
        # entries: [(name, dtype, shape, nbytes)] in physical write order
        header: dict = {"__metadata__": dict(metadata)}
        offset = 0
        seen = set()
        for name, dtype, shape, nbytes in entries:
            if name in seen:
                raise ValueError(f"duplicate tensor name {name}")
            seen.add(name)
            if dtype not in ITEMSIZE:
                raise ValueError(f"unsupported dtype {dtype}")
            expected = _numel(shape) * ITEMSIZE[dtype]
            if nbytes != expected:
                raise ValueError(f"{name}: {nbytes} bytes declared, {expected} implied by shape and dtype")
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw += b" " * ((8 - len(raw) % 8) % 8)  # keep the data section 8-byte aligned, as safetensors does
        self._entries = list(entries)
        self._index = 0
        self._left = self._entries[0][3] if self._entries else 0
        self._path = path
        self._tmp = path + ".partial"
        self._file = open(self._tmp, "wb")
        self._sha = hashlib.sha256()
        self._write(struct.pack("<Q", len(raw)) + raw)
        self.header_bytes = len(raw) + 8
        self.data_bytes = offset

    def _write(self, data) -> None:
        self._file.write(data)
        self._sha.update(data)

    def write(self, name: str, data) -> None:
        """Append bytes for the next declared tensor. data: bytes-like with the exact length."""
        if self._index >= len(self._entries):
            raise ValueError("more data written than declared")
        expect_name = self._entries[self._index][0]
        if name != expect_name:
            raise ValueError(f"wrote {name} where {expect_name} was declared")
        view = memoryview(data).cast("B")
        if view.nbytes > self._left:
            raise ValueError(f"{name}: more bytes than declared")
        self._write(view)
        self._left -= view.nbytes
        if self._left == 0:
            self._index += 1
            if self._index < len(self._entries):
                self._left = self._entries[self._index][3]

    def finish(self) -> str:
        """Check that every declared byte was written, make it durable and move it into place.
        Returns the sha256 of the complete file."""
        if self._index != len(self._entries) or self._left != 0:
            self._file.close()
            os.remove(self._tmp)
            raise ValueError("checkpoint writer finished before all declared bytes were written")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self._tmp, self._path)
        return self._sha.hexdigest()

    def abort(self) -> None:
        try:
            self._file.close()
        finally:
            if os.path.exists(self._tmp):
                os.remove(self._tmp)


def sha256_file(path: str, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


