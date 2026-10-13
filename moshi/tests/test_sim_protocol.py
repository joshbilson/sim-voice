# SPDX-License-Identifier: MIT
"""Unit tests for moshi.sim_protocol: validation, bearer auth, token file, codec, buffers."""

import json
import os
import tempfile
import unittest

import numpy as np

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
from moshi import sim_protocol as P

VOICES = ["NATF0.pt", "NATF2.pt"]


class ChatQueryTests(unittest.TestCase):
    def parse(self, **query):
        return P.parse_chat_query(query, voices=VOICES, default_voice="NATF2.pt",
                                  default_text_prompt="DEFAULT")

    def test_defaults_when_absent(self):
        params = self.parse()
        self.assertEqual(params.codec, "opus")
        self.assertEqual(params.voice_prompt, "NATF2.pt")
        self.assertEqual(params.text_prompt, "DEFAULT")
        self.assertIsNone(params.seed)
        self.assertRegex(params.session, r"^[0-9a-f]{16}$")

    def test_empty_values_mean_absent(self):
        params = self.parse(codec="", voice_prompt="", text_prompt="   ", seed="", session="")
        self.assertEqual((params.codec, params.voice_prompt, params.text_prompt), ("opus", "NATF2.pt", "DEFAULT"))
        self.assertIsNone(params.seed)

    def test_voice_must_match_pattern_and_exist(self):
        self.assertEqual(self.parse(voice_prompt="NATF0.pt").voice_prompt, "NATF0.pt")
        for bad in ["../NATF2.pt", "a/b.pt", "NATF2.wav", "NATF9.pt", "NATF2.pt\n", "NATF2.pt.pt",
                    "x" * 41 + ".pt", ".pt", "NATF 2.pt"]:
            with self.subTest(voice=bad), self.assertRaises(P.BadRequest):
                self.parse(voice_prompt=bad)

    def test_text_prompt_limit_is_1200_characters(self):
        self.assertEqual(len(self.parse(text_prompt="a" * 1200).text_prompt), 1200)
        with self.assertRaises(P.BadRequest) as ctx:
            self.parse(text_prompt="a" * 1201)
        self.assertNotIn("aaaa", str(ctx.exception))  # never echoes user input

    def test_codec_values(self):
        self.assertEqual(self.parse(codec="pcm16").codec, "pcm16")
        with self.assertRaises(P.BadRequest):
            self.parse(codec="mp3")

    def test_seed_rules(self):
        self.assertIsNone(P.parse_seed("-1"))
        self.assertEqual(P.parse_seed("0"), 0)
        self.assertEqual(P.parse_seed("42"), 42)
        self.assertEqual(P.parse_seed(str(2**32 - 1)), 2**32 - 1)
        for bad in ["abc", str(2**32), "-2", "1.5"]:
            with self.subTest(seed=bad), self.assertRaises(P.BadRequest):
                P.parse_seed(bad)

    def test_session_rules(self):
        self.assertEqual(P.validate_session("call-42"), "call-42")
        self.assertEqual(P.validate_session("a" * 64), "a" * 64)
        for bad in ["has space", "a/b", "a" * 65, "abc\n", "under_score"]:
            with self.subTest(session=bad), self.assertRaises(P.BadRequest):
                P.validate_session(bad)


class AuthTests(unittest.TestCase):
    token = b"test-token-0123456789abcdef"

    def test_bearer_accepts_only_the_exact_token(self):
        self.assertTrue(P.bearer_matches("Bearer " + self.token.decode(), self.token))
        self.assertTrue(P.bearer_matches("bearer " + self.token.decode(), self.token))  # scheme is case-insensitive
        self.assertFalse(P.bearer_matches("Bearer test-token-0123456789abcdeX", self.token))
        self.assertFalse(P.bearer_matches("Bearer " + self.token.decode()[:-1], self.token))
        self.assertFalse(P.bearer_matches(self.token.decode(), self.token))  # no scheme
        self.assertFalse(P.bearer_matches("Basic " + self.token.decode(), self.token))
        self.assertFalse(P.bearer_matches("Bearer ", self.token))
        self.assertFalse(P.bearer_matches(None, self.token))
        self.assertFalse(P.bearer_matches("", self.token))

    def test_empty_expectation_never_matches(self):
        self.assertFalse(P.bearer_matches("Bearer ", b""))

    def test_non_ascii_header_is_rejected_not_raised(self):
        self.assertFalse(P.bearer_matches("Bearer tøken-ünïcode-value-0123", self.token))

    def test_loopback_detection(self):
        for host in ["localhost", "127.0.0.1", "127.8.9.10", "::1", "[::1]", "LOCALHOST"]:
            with self.subTest(host=host):
                self.assertTrue(P.is_loopback_host(host))
        for host in ["0.0.0.0", "::", "192.0.2.10", "example.org", "", "10.0.0.1"]:
            with self.subTest(host=host):
                self.assertFalse(P.is_loopback_host(host))


class TokenFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "token")

    def write(self, text, mode):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(self.path, mode)

    def test_private_token_loads_and_is_stripped(self):
        self.write("  abcdefghijklmnop0123\n", 0o600)
        self.assertEqual(P.load_token_file(self.path), "abcdefghijklmnop0123")

    def test_group_or_world_readable_is_refused(self):
        for mode in (0o640, 0o604, 0o644):
            with self.subTest(mode=oct(mode)):
                self.write("abcdefghijklmnop0123", mode)
                with self.assertRaises(P.TokenFileError) as ctx:
                    P.load_token_file(self.path)
                self.assertNotIn("abcdefghijklmnop0123", str(ctx.exception))

    def test_short_or_spaced_token_is_refused(self):
        self.write("short", 0o600)
        with self.assertRaises(P.TokenFileError):
            P.load_token_file(self.path)
        self.write("abcdefghijklmnop 0123", 0o600)
        with self.assertRaises(P.TokenFileError):
            P.load_token_file(self.path)

    def test_missing_or_directory_is_refused(self):
        with self.assertRaises(P.TokenFileError):
            P.load_token_file(os.path.join(self.dir.name, "absent"))
        with self.assertRaises(P.TokenFileError):
            P.load_token_file(self.dir.name)


class DefaultTextPromptTests(unittest.TestCase):
    def test_none_gives_builtin_default(self):
        self.assertEqual(P.read_default_text_prompt(None), "You enjoy having a good conversation.")

    def test_file_contents_stripped_and_empty_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "prompt.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("  Be brief.\n")
            self.assertEqual(P.read_default_text_prompt(path), "Be brief.")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("   \n")
            self.assertEqual(P.read_default_text_prompt(path), "You enjoy having a good conversation.")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("x" * 1201)
            with self.assertRaises(ValueError):
                P.read_default_text_prompt(path)


class Pcm16CodecTests(unittest.TestCase):
    def test_round_trip_is_exact(self):
        rng = np.random.default_rng(7)
        samples = rng.integers(-32768, 32768, size=1920, dtype=np.int32).astype("<i2")
        codec = P.Pcm16Codec()
        pcm = codec.decode(samples.tobytes())
        self.assertEqual(pcm.dtype, np.float32)
        self.assertTrue(np.all(np.abs(pcm) <= 1.0))
        self.assertEqual(codec.encode(pcm), samples.tobytes())

    def test_odd_trailing_byte_is_carried_to_the_next_message(self):
        codec = P.Pcm16Codec()
        first = codec.decode(b"\x01\x00\x02")          # sample 1, plus one byte of sample 2
        self.assertEqual(first.size, 1)
        second = codec.decode(b"\x00\x03\x00")          # completes sample 2, then sample 3
        self.assertEqual(codec.encode(second), b"\x02\x00\x03\x00")
        self.assertEqual(codec.decode(b"\x04").size, 0)  # half a sample: nothing yet

    def test_encode_clips_and_handles_nan(self):
        codec = P.Pcm16Codec()
        out = codec.encode(np.array([2.0, -2.0, 0.0, np.nan], dtype=np.float32))
        self.assertEqual(np.frombuffer(out, "<i2").tolist(), [32767, -32768, 0, 0])


class FrameBufferTests(unittest.TestCase):
    def test_partial_frames_wait_for_completion(self):
        buffer = P.FrameBuffer(frame_samples=4, max_frames=6)
        self.assertEqual(buffer.push(np.arange(3, dtype=np.float32)), 0)
        self.assertEqual(len(buffer), 0)
        buffer.push(np.arange(3, 6, dtype=np.float32))
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.pop().tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertIsNone(buffer.pop())

    def test_oldest_frames_are_dropped_beyond_the_backlog(self):
        buffer = P.FrameBuffer(frame_samples=2, max_frames=3)
        dropped = buffer.push(np.arange(10, dtype=np.float32))  # five frames
        self.assertEqual(dropped, 2)
        self.assertEqual(buffer.dropped, 2)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.pop().tolist(), [4.0, 5.0])  # the newest three survive, oldest first
        self.assertEqual(buffer.pop().tolist(), [6.0, 7.0])
        self.assertEqual(buffer.pop().tolist(), [8.0, 9.0])

    def test_rejects_zero_backlog(self):
        with self.assertRaises(ValueError):
            P.FrameBuffer(max_frames=0)


class LatencyWindowTests(unittest.TestCase):
    def test_median_of_recent_frames(self):
        window = P.LatencyWindow(size=3)
        self.assertIsNone(window.p50())
        for value in (10.0, 1.0, 2.0, 3.0):  # the first sample falls out of the window
            window.add(value)
        self.assertEqual(window.p50(), 2.0)


class JsonMessageTests(unittest.TestCase):
    def test_kind_byte_then_compact_json(self):
        message = P.json_message(P.KIND_STATS, {"type": "stats", "backlogFrames": 0})
        self.assertEqual(message[0], 0x06)
        self.assertEqual(json.loads(message[1:]), {"type": "stats", "backlogFrames": 0})
        self.assertNotIn(b" ", message)

    def test_error_message_shape(self):
        message = P.error_message("busy", "a call is already in progress")
        self.assertEqual(message[0], 0x05)
        self.assertEqual(json.loads(message[1:]), {"code": "busy", "message": "a call is already in progress"})


if __name__ == "__main__":
    unittest.main()
