# SPDX-License-Identifier: MIT
"""Start-up rules of the Sim voice server (moshi.sim_cli): defaults and refusals."""

import contextlib
import io
import os
import tempfile
import unittest

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
from moshi.sim_cli import build_arg_parser, validate_serve_args

TOKEN = "cli-token-0123456789abcdef"


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.token_path = os.path.join(self.dir.name, "token")
        with open(self.token_path, "w", encoding="utf-8") as handle:
            handle.write(TOKEN + "\n")
        os.chmod(self.token_path, 0o600)
        self.parser = build_arg_parser()

    def validate(self, *argv):
        args = self.parser.parse_args(list(argv))
        return validate_serve_args(self.parser, args)

    def refused(self, *argv):
        """Parse and validate; return the message the parser printed when it refused."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            self.validate(*argv)
        self.assertEqual(ctx.exception.code, 2)
        return stderr.getvalue()

    def test_defaults_match_the_contract(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.host, "localhost")
        self.assertEqual(args.port, 8998)
        self.assertEqual(args.static, "none")
        self.assertEqual(args.default_voice, "NATF2.pt")
        self.assertEqual(args.max_backlog_frames, 6)
        self.assertEqual(args.idle_timeout_s, 15.0)
        self.assertEqual(args.max_session_seconds, 1800.0)
        self.assertEqual(args.stats_interval_s, 2.0)
        self.assertIsNone(args.auth_token_file)
        self.assertEqual(args.hf_repo, "nvidia/personaplex-7b-v1")

    def test_loopback_without_a_token_is_allowed_and_means_no_auth(self):
        token, prompt = self.validate("--host", "127.0.0.1")
        self.assertIsNone(token)
        self.assertEqual(prompt, "You enjoy having a good conversation.")

    def test_non_loopback_host_needs_a_token_file(self):
        message = self.refused("--host", "0.0.0.0")
        self.assertIn("--auth-token-file", message)
        self.assertNotIn(TOKEN, message)

    def test_non_loopback_host_with_a_token_file_starts(self):
        token, _ = self.validate("--host", "192.0.2.10", "--auth-token-file", self.token_path)
        self.assertEqual(token, TOKEN)

    def test_token_file_must_be_private(self):
        os.chmod(self.token_path, 0o644)
        message = self.refused("--host", "192.0.2.10", "--auth-token-file", self.token_path)
        self.assertIn("chmod 600", message)
        self.assertNotIn(TOKEN, message)

    def test_missing_token_file_is_refused(self):
        self.refused("--auth-token-file", os.path.join(self.dir.name, "absent"))

    def test_gradio_tunnel_flags_are_refused(self):
        self.assertIn("removed", self.refused("--gradio-tunnel"))
        self.assertIn("removed", self.refused("--gradio-tunnel-token", "abc"))

    def test_fp8_and_w8a16_are_exclusive(self):
        self.refused("--fp8", "--w8a16")

    def test_limits_must_be_sane(self):
        self.refused("--max-backlog-frames", "0")
        self.refused("--idle-timeout-s", "0")
        self.refused("--max-session-seconds", "-5")
        self.refused("--stats-interval-s", "0")

    def test_default_voice_must_look_like_a_voice_file(self):
        self.refused("--default-voice", "../NATF2.pt")
        self.refused("--default-voice", "NATF2.wav")

    def test_custom_voices_are_off_by_default(self):
        self.assertIsNone(self.parser.parse_args([]).custom_voice_dir)

    def test_custom_voice_dir_may_be_missing_or_private_but_not_open(self):
        path = os.path.join(self.dir.name, "voices-custom")
        self.validate("--custom-voice-dir", path)  # missing: created 0700 when the server starts
        os.mkdir(path, 0o700)
        self.validate("--custom-voice-dir", path)
        os.chmod(path, 0o755)
        self.assertIn("group or world", self.refused("--custom-voice-dir", path))
        plain = os.path.join(self.dir.name, "plain")
        open(plain, "w").close()
        self.assertIn("not a directory", self.refused("--custom-voice-dir", plain))
        self.assertIn("parent", self.refused("--custom-voice-dir", os.path.join(self.dir.name, "x", "y")))

    def test_a_custom_voice_cannot_be_the_default_voice(self):
        self.refused("--default-voice", "c-warm-sim.pt")

    def test_default_text_prompt_file_is_read_and_limited(self):
        path = os.path.join(self.dir.name, "prompt.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("  Speak slowly.  ")
        _, prompt = self.validate("--default-text-prompt-file", path)
        self.assertEqual(prompt, "Speak slowly.")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x" * 1201)
        self.refused("--default-text-prompt-file", path)
        self.refused("--default-text-prompt-file", os.path.join(self.dir.name, "absent.txt"))




class W8a16CheckpointFlag(unittest.TestCase):
    """--w8a16-checkpoint: refusals that need no torch (the file header is read, never the data)."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.parser = build_arg_parser()

    def checkpoint(self, name, version="1"):
        from moshi.w8a16_format import METADATA_PREFIX, SafetensorsWriter
        path = os.path.join(self.dir.name, name)
        writer = SafetensorsWriter(path, [("x", "U8", (1,), 1)], {METADATA_PREFIX + "version": version})
        writer.write("x", b"\x00")
        writer.finish()
        return path

    def refused(self, *argv):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            validate_serve_args(self.parser, self.parser.parse_args(list(argv)))
        self.assertEqual(ctx.exception.code, 2)
        return stderr.getvalue()

    def test_a_converted_file_is_accepted(self):
        path = self.checkpoint("ok.safetensors")
        validate_serve_args(self.parser, self.parser.parse_args(["--host", "127.0.0.1", "--w8a16-checkpoint", path]))

    def test_another_layout_version_is_refused(self):
        path = self.checkpoint("old.safetensors", version="0")
        self.assertIn("not a w8a16 checkpoint", self.refused("--host", "127.0.0.1", "--w8a16-checkpoint", path))

    def test_a_missing_or_corrupt_file_is_refused(self):
        self.assertIn("cannot read", self.refused("--host", "127.0.0.1", "--w8a16-checkpoint",
                                                  os.path.join(self.dir.name, "missing.safetensors")))
        bad = os.path.join(self.dir.name, "bad.safetensors")
        with open(bad, "wb") as handle:
            handle.write(b"not a checkpoint at all")
        self.refused("--host", "127.0.0.1", "--w8a16-checkpoint", bad)

    def test_it_excludes_the_other_weight_sources_and_modes(self):
        path = self.checkpoint("ok.safetensors")
        self.assertIn("not both", self.refused("--host", "127.0.0.1", "--w8a16-checkpoint", path,
                                               "--moshi-weight", "/x/model.safetensors"))
        self.assertIn("mutually exclusive", self.refused("--host", "127.0.0.1", "--w8a16-checkpoint", path,
                                                         "--fp8"))
        self.assertIn("--cpu-offload", self.refused("--host", "127.0.0.1", "--w8a16-checkpoint", path,
                                                    "--cpu-offload"))


if __name__ == "__main__":
    unittest.main()
