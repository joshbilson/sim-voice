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


if __name__ == "__main__":
    unittest.main()
