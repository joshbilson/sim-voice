# SPDX-License-Identifier: MIT
"""Torch-free safetensors reader and streaming writer used by the w8a16 checkpoint."""

import json
import os
import struct
import tempfile
import unittest

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)

from moshi.w8a16_format import (  # noqa: E402
    CONVERSION_VERSION,
    METADATA_PREFIX,
    CheckpointFormatError,
    SafetensorsWriter,
    check_w8a16_metadata,
    read_header,
    sha256_file,
)


def write_file(path, entries, payloads, metadata=None):
    writer = SafetensorsWriter(path, entries, metadata or {})
    for (name, _, _, _), data in zip(entries, payloads):
        writer.write(name, data)
    return writer.finish()


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "t.safetensors")

    def tearDown(self):
        self.dir.cleanup()

    def test_layout_metadata_and_checksum(self):
        entries = [
            ("a.weight", "F8_E4M3", (2, 3), 6),
            ("a.scale", "F32", (2,), 8),
            ("b.empty", "BF16", (0,), 0),
            ("c.bf16", "BF16", (1, 2), 4),
        ]
        payloads = [bytes(range(6)), struct.pack("<2f", 0.5, 2.0), b"", b"\x01\x02\x03\x04"]
        digest = write_file(self.path, entries, payloads, {"k": "v", "format": "pt"})
        self.assertEqual(digest, sha256_file(self.path))

        header = read_header(self.path)
        self.assertEqual(header.metadata, {"k": "v", "format": "pt"})
        self.assertEqual(list(header.tensors), ["a.weight", "a.scale", "b.empty", "c.bf16"])
        self.assertEqual(header.tensors["a.weight"].dtype, "F8_E4M3")
        self.assertEqual(header.tensors["a.weight"].shape, (2, 3))
        self.assertEqual(header.tensors["b.empty"].nbytes, 0)
        self.assertEqual(header.tensors["c.bf16"].begin, 14)
        self.assertEqual(header.data_start % 8, 0, "data section must be 8-byte aligned")
        with open(self.path, "rb") as handle:
            raw = handle.read()
        self.assertEqual(raw[header.data_start:header.data_start + 6], bytes(range(6)))
        self.assertEqual(raw[-4:], b"\x01\x02\x03\x04")

    def test_header_is_json_with_the_safetensors_offsets(self):
        entries = [("x", "U8", (3,), 3)]
        write_file(self.path, entries, [b"abc"])
        with open(self.path, "rb") as handle:
            (n,) = struct.unpack("<Q", handle.read(8))
            obj = json.loads(handle.read(n))
        self.assertEqual(obj["x"], {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]})


class WriterRefusals(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "t.safetensors")

    def tearDown(self):
        self.dir.cleanup()

    def test_declared_size_must_match_shape_and_dtype(self):
        with self.assertRaises(ValueError):
            SafetensorsWriter(self.path, [("x", "BF16", (2, 2), 7)], {})

    def test_unknown_dtype_and_duplicate_names_are_refused(self):
        with self.assertRaises(ValueError):
            SafetensorsWriter(self.path, [("x", "C64", (1,), 8)], {})
        with self.assertRaises(ValueError):
            SafetensorsWriter(self.path, [("x", "U8", (1,), 1), ("x", "U8", (1,), 1)], {})

    def test_wrong_name_or_length_is_refused(self):
        writer = SafetensorsWriter(self.path, [("x", "U8", (2,), 2)], {})
        with self.assertRaises(ValueError):
            writer.write("y", b"ab")
        with self.assertRaises(ValueError):
            writer.write("x", b"abc")
        writer.abort()
        self.assertFalse(os.path.exists(self.path + ".partial"))

    def test_finish_before_all_bytes_removes_the_partial_file(self):
        writer = SafetensorsWriter(self.path, [("x", "U8", (4,), 4)], {})
        writer.write("x", b"ab")
        with self.assertRaises(ValueError):
            writer.finish()
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.path + ".partial"))

    def test_partial_write_is_renamed_only_on_finish(self):
        writer = SafetensorsWriter(self.path, [("x", "U8", (4,), 4)], {})
        writer.write("x", b"ab")
        self.assertTrue(os.path.exists(self.path + ".partial"))
        self.assertFalse(os.path.exists(self.path))
        writer.write("x", b"cd")
        writer.finish()
        self.assertTrue(os.path.exists(self.path))
        self.assertFalse(os.path.exists(self.path + ".partial"))


class ReaderRefusals(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "t.safetensors")

    def tearDown(self):
        self.dir.cleanup()

    def _raw(self, header_obj, body=b""):
        raw = json.dumps(header_obj).encode("utf-8")
        with open(self.path, "wb") as handle:
            handle.write(struct.pack("<Q", len(raw)) + raw + body)

    def test_truncated_and_empty_files(self):
        with open(self.path, "wb") as handle:
            handle.write(b"\x01\x00")
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)
        self.assertRaises(CheckpointFormatError, read_header, os.path.join(self.dir.name, "missing"))

    def test_bad_json_and_non_object_headers(self):
        with open(self.path, "wb") as handle:
            handle.write(struct.pack("<Q", 5) + b"{oops")
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)
        self._raw([1, 2, 3])
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)

    def test_offsets_must_be_contiguous_and_fill_the_file(self):
        self._raw({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                   "b": {"dtype": "U8", "shape": [2], "data_offsets": [3, 5]}}, b"xxxxx")
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)
        self._raw({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}, b"xxx")
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)

    def test_size_must_match_shape_and_dtype(self):
        self._raw({"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 2]}}, b"xx")
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)

    def test_unsupported_dtype_and_bad_metadata(self):
        self._raw({"a": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}}, b"x" * 8)
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)
        self._raw({"__metadata__": {"n": 3}, "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, b"x")
        with self.assertRaises(CheckpointFormatError):
            read_header(self.path)


class MetadataGate(unittest.TestCase):
    def test_current_version_is_accepted(self):
        check_w8a16_metadata({METADATA_PREFIX + "version": CONVERSION_VERSION})

    def test_other_or_missing_versions_are_refused(self):
        with self.assertRaises(CheckpointFormatError):
            check_w8a16_metadata({METADATA_PREFIX + "version": "0"})
        with self.assertRaises(CheckpointFormatError):
            check_w8a16_metadata({})


if __name__ == "__main__":
    unittest.main()
