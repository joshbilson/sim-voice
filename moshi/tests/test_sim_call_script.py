# SPDX-License-Identifier: MIT
"""End to end: scripts/sim_call_test.py against a fake-engine server on a real port (no GPU)."""

import asyncio
import json
import logging
import os
import sys
import tempfile
import unittest
import wave

import aiohttp
import numpy as np
from aiohttp import web
from aiohttp.test_utils import unused_port

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
from fake_engine import FakeEngine
from moshi.sim_session import SessionConfig, create_app

SCRIPT = os.path.join(_sim_bootstrap.REPO_ROOT, "scripts", "sim_call_test.py")
TOKEN = "call-script-token-0123456789"


def write_wav(path, seconds, rate=24000, channels=1):
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype("<i2")
    if channels > 1:
        tone = np.repeat(tone[:, None], channels, axis=1).reshape(-1)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(tone.tobytes())


def write_token(path, text, mode=0o600):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    os.chmod(path, mode)


class CallScriptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        logging.getLogger("asyncio").setLevel(logging.ERROR)
        logging.getLogger("moshi.sim_session").setLevel(logging.CRITICAL)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.token_path = os.path.join(self.dir.name, "token")
        write_token(self.token_path, TOKEN)
        self.wav = os.path.join(self.dir.name, "in.wav")
        write_wav(self.wav, 0.5)

        self.engine = FakeEngine()
        runner = web.AppRunner(create_app(self.engine, SessionConfig(auth_token=TOKEN, stats_interval_s=0.2)),
                               access_log=None)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        self.port = unused_port()
        await web.TCPSite(runner, "127.0.0.1", self.port).start()
        self.url = f"http://127.0.0.1:{self.port}"

    async def run_script(self, *args):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, SCRIPT, *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
        return proc.returncode, out.decode("utf-8"), err.decode("utf-8")

    async def test_full_call_records_audio_and_text_and_summarises(self):
        out_wav = os.path.join(self.dir.name, "out.wav")
        out_text = os.path.join(self.dir.name, "out.txt")
        rc, out, err = await self.run_script(
            "--url", self.url, "--token-file", self.token_path, "--wav", self.wav,
            "--voice", "NATF0.pt", "--text-prompt", "Be kind.", "--seconds", "1.0",
            "--tail-seconds", "0.3", "--out-wav", out_wav, "--out-text", out_text)
        self.assertEqual(rc, 0, err)
        summary = json.loads(out)
        self.assertIs(summary["ok"], True)
        self.assertEqual(summary["closeCode"], 1000)
        self.assertIsNone(summary["errorCode"])
        self.assertEqual(summary["accepted"]["voice"], "NATF0.pt")
        self.assertEqual(summary["accepted"]["codec"], "pcm16")
        self.assertEqual(summary["framesIn"], 13)  # ceil(1.0 s / 80 ms)
        self.assertEqual(summary["framesOut"], summary["framesIn"])  # the fake engine echoes every frame
        self.assertEqual(summary["textPieces"], summary["framesOut"])
        for key in ("msToHandshake", "msToAccepted", "msToLive", "msToFirstAudio"):
            self.assertIsInstance(summary[key], float, key)
        self.assertEqual(self.engine.prompts[-1]["voice_prompt"], "NATF0.pt")
        self.assertEqual(self.engine.prompts[-1]["text_prompt"], "Be kind.")

        with wave.open(out_wav, "rb") as handle:
            self.assertEqual((handle.getnchannels(), handle.getsampwidth(), handle.getframerate()), (1, 2, 24000))
            self.assertEqual(handle.getnframes(), summary["framesOut"] * 1920)
        with open(out_text, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "a" * summary["textPieces"])

        # The token and the persona text never reach the terminal.
        self.assertNotIn(TOKEN, out + err)
        self.assertNotIn("Be kind.", out + err)

    async def test_wrong_token_is_refused_with_exit_2(self):
        bad = os.path.join(self.dir.name, "bad-token")
        write_token(bad, "wrong-token-0123456789abcd")
        rc, out, err = await self.run_script("--url", self.url, "--token-file", bad, "--wav", self.wav)
        self.assertEqual(rc, 2)
        summary = json.loads(out)
        self.assertEqual(summary["error"], "handshake_refused")
        self.assertEqual(summary["status"], 401)
        self.assertNotIn("wrong-token", out + err)

    async def test_busy_server_gives_error_code_and_exit_1(self):
        async with aiohttp.ClientSession() as http:
            holder = await http.ws_connect(f"ws://127.0.0.1:{self.port}/api/chat?codec=pcm16",
                                           headers={"Authorization": f"Bearer {TOKEN}"})
            try:
                first = await holder.receive(timeout=5)
                self.assertEqual(first.data[0], 0x04)
                handshake = await holder.receive(timeout=5)
                self.assertEqual(handshake.data[0], 0x00)

                rc, out, err = await self.run_script("--url", self.url, "--token-file", self.token_path,
                                                     "--wav", self.wav)
                self.assertEqual(rc, 1, err)
                summary = json.loads(out)
                self.assertIs(summary["ok"], False)
                self.assertEqual(summary["errorCode"], "busy")
                self.assertEqual(summary["closeCode"], 4409)
            finally:
                await holder.close()

    async def test_token_file_must_be_private(self):
        loose = os.path.join(self.dir.name, "loose-token")
        write_token(loose, TOKEN, mode=0o644)
        rc, out, err = await self.run_script("--url", self.url, "--token-file", loose)
        self.assertEqual(rc, 2)
        self.assertIn("chmod 600", err)
        self.assertNotIn(TOKEN, out + err)

    async def test_stereo_48k_input_is_downmixed_and_resampled(self):
        stereo = os.path.join(self.dir.name, "stereo48.wav")
        write_wav(stereo, 0.5, rate=48000, channels=2)
        rc, out, err = await self.run_script("--url", self.url, "--token-file", self.token_path,
                                             "--wav", stereo, "--seconds", "0.5", "--tail-seconds", "0.2")
        self.assertEqual(rc, 0, err)
        summary = json.loads(out)
        self.assertEqual(summary["framesIn"], 7)  # ceil(0.5 s / 80 ms)
        self.assertIs(summary["ok"], True)


if __name__ == "__main__":
    unittest.main()
