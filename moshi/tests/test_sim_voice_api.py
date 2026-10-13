# SPDX-License-Identifier: MIT
"""Custom voices over HTTP (PUT/GET/DELETE /api/voices/<id>, health, busy gate). Fake engine, no torch."""

import asyncio
import io
import json
import logging
import os
import stat
import tempfile
import unittest
import wave
from urllib.parse import urlencode

import aiohttp
from aiohttp import web
from aiohttp.test_utils import unused_port

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
import voice_fixtures as F
from fake_engine import FakeEngine
from moshi.sim_session import SessionConfig, create_app
from moshi.sim_voices import UPLOAD_LIMIT_BYTES
from test_sim_session import TOKEN, drain_until_closed, next_binary

logging.getLogger("asyncio").setLevel(logging.ERROR)
SESSION_LOGGER = "moshi.sim_session"
VOICE = "c-test-service"


def speech_upload(seconds: float = 4.5, amplitude: float = 0.3) -> bytes:
    return F.pcm_wav_bytes(F.speech_like(seconds, amplitude))


class VoiceApiTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        logger = logging.getLogger(SESSION_LOGGER)
        previous = logger.level
        logger.setLevel(logging.CRITICAL)
        self.addCleanup(logger.setLevel, previous)
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.custom = os.path.join(self.root.name, "voices-custom")
        self.http = aiohttp.ClientSession()
        self.addAsyncCleanup(self.http.close)
        self.engine = None

    async def start(self, engine=None, custom=True, **config):
        directory = self.custom if custom else None
        self.engine = engine or FakeEngine(custom_voice_dir=directory)
        self.cfg = SessionConfig(auth_token=TOKEN, custom_voice_dir=directory, **config)
        runner = web.AppRunner(create_app(self.engine, self.cfg), handler_cancellation=False, access_log=None)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        port = unused_port()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        self.base = f"http://127.0.0.1:{port}"

    def auth(self):
        return {"Authorization": f"Bearer {TOKEN}"}

    async def send(self, method, path, body=None, *, auth=True, headers=None):
        """(status, response headers, body bytes)."""
        merged = dict(self.auth() if auth else {})
        merged.update(headers or {})
        if isinstance(body, bytes) and len(body) > 64 * 1024:
            body = io.BytesIO(body)  # streamed, so aiohttp does not warn about a large raw body
        async with self.http.request(method, f"{self.base}{path}", data=body, headers=merged) as resp:
            return resp.status, resp.headers, await resp.read()

    async def put(self, voice, body, **kwargs):
        status, _, data = await self.send("PUT", f"/api/voices/{voice}", body,
                                          headers={"Content-Type": "audio/wav"}, **kwargs)
        return status, (json.loads(data) if data else None)

    async def health(self):
        _, _, data = await self.send("GET", "/api/health")
        return json.loads(data)

    async def call(self, **params):
        query = urlencode({"codec": "pcm16", **params})
        return await self.http.ws_connect(f"ws://{self.base.split('://', 1)[1]}/api/chat?{query}",
                                          headers=self.auth())

    async def wait_not_busy(self, timeout=5.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while (await self.health())["busy"]:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("still busy")
            await asyncio.sleep(0.02)


class AccessAndOffTests(VoiceApiTestCase):
    async def test_every_voice_route_needs_the_bearer_token(self):
        await self.start()
        for method, path in (("PUT", f"/api/voices/{VOICE}"), ("DELETE", f"/api/voices/{VOICE}"),
                             ("GET", f"/api/voices/{VOICE}.wav")):
            for headers in ({}, {"Authorization": "Bearer wrong-token-value-0000"}):
                with self.subTest(method=method, headers=sorted(headers)):
                    status, _, _ = await self.send(method, path, speech_upload(), auth=False, headers=headers)
                    self.assertEqual(status, 401)
        self.assertFalse(os.path.exists(self.custom) and os.listdir(self.custom))

    async def test_without_a_custom_directory_the_routes_are_404_and_health_is_presets_only(self):
        await self.start(custom=False)
        status, body = await self.put(VOICE, speech_upload())
        self.assertEqual((status, body["code"]), (404, "not_found"))
        status, _, _ = await self.send("DELETE", f"/api/voices/{VOICE}")
        self.assertEqual(status, 404)
        status, _, _ = await self.send("GET", f"/api/voices/{VOICE}.wav")
        self.assertEqual(status, 404)
        health = await self.health()
        self.assertEqual(health["voices"], ["NATF0.pt", "NATF2.pt"])
        self.assertEqual(health["customVoices"], [])


class UploadTests(VoiceApiTestCase):
    async def test_a_sample_builds_both_files_privately(self):
        await self.start()
        status, body = await self.put(VOICE, speech_upload())
        self.assertEqual(status, 201)
        self.assertEqual(set(body), {"voice", "seconds", "frames", "ms"})
        self.assertEqual((body["voice"], body["seconds"], body["frames"]), ("c-test-service.pt", 4.0, 50))
        self.assertIsInstance(body["ms"], int)
        pt = os.path.join(self.custom, f"{VOICE}.pt")
        wav = os.path.join(self.custom, f"{VOICE}.wav")
        with open(pt, "rb") as handle:
            self.assertEqual(handle.read(), b"FAKE-VOICE-PROMPT")
        with wave.open(wav, "rb") as sample:
            self.assertEqual((sample.getnchannels(), sample.getsampwidth(), sample.getframerate()), (1, 2, 24000))
            self.assertEqual(sample.getnframes(), 96000)
        self.assertEqual(stat.S_IMODE(os.stat(pt).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(wav).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.custom).st_mode), 0o700)
        self.assertEqual([n for n in os.listdir(self.custom) if n.startswith(".build-")], [])

    async def test_health_lists_presets_then_custom_voices(self):
        await self.start()
        await self.put(VOICE, speech_upload())
        health = await self.health()
        self.assertEqual(health["voices"], ["NATF0.pt", "NATF2.pt", "c-test-service.pt"])
        self.assertEqual(health["customVoices"], ["c-test-service.pt"])

    async def test_the_sample_is_served_back_as_wav(self):
        await self.start()
        await self.put(VOICE, speech_upload())
        with open(os.path.join(self.custom, f"{VOICE}.wav"), "rb") as handle:
            expected = handle.read()
        status, headers, data = await self.send("GET", f"/api/voices/{VOICE}.wav")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        self.assertEqual(data, expected)

    async def test_overwrite_replaces_the_files(self):
        await self.start()
        await self.put(VOICE, speech_upload(4.5, 0.3))
        with open(os.path.join(self.custom, f"{VOICE}.wav"), "rb") as handle:
            first = handle.read()
        status, _ = await self.put(VOICE, speech_upload(5.0, 0.1))
        self.assertEqual(status, 201)
        with open(os.path.join(self.custom, f"{VOICE}.wav"), "rb") as handle:
            self.assertNotEqual(handle.read(), first)

    async def test_an_upload_of_exactly_four_mebibytes_is_accepted(self):
        await self.start()
        body = speech_upload()
        body += b"\x00" * (UPLOAD_LIMIT_BYTES - len(body))  # trailing bytes after the data chunk
        self.assertEqual(len(body), UPLOAD_LIMIT_BYTES)
        status, _ = await self.put(VOICE, body)
        self.assertEqual(status, 201)

    async def test_bigger_uploads_are_413_declared_or_streamed(self):
        await self.start()
        too_big = b"\x00" * (UPLOAD_LIMIT_BYTES + 1)
        status, body = await self.put(VOICE, too_big)
        self.assertEqual((status, body["code"]), (413, "too_large"))

        async def chunks():
            yield b"\x00" * (UPLOAD_LIMIT_BYTES + 1)  # no Content-Length: a chunked upload, all read
        try:
            status, _, data = await self.send("PUT", f"/api/voices/{VOICE}", chunks(),
                                              headers={"Content-Type": "audio/wav"})
            self.assertEqual(status, 413)
            self.assertEqual(json.loads(data)["code"], "too_large")
        except aiohttp.ClientError:
            pass  # the server may close while the client is still sending: nothing was built either way
        self.assertFalse(os.path.exists(os.path.join(self.custom, f"{VOICE}.pt")))

    async def test_unusable_audio_is_400_bad_audio_and_installs_nothing(self):
        await self.start()
        cases = {
            "not a wav": b"hello world, not audio",
            "empty body": b"",
            "silence": F.pcm_wav_bytes(F.silence(5)),
            "1.5 s of speech": F.pcm_wav_bytes(F.concat(F.speech_like(1.5, 0.3), F.silence(4))),
        }
        for label, body in cases.items():
            with self.subTest(case=label):
                status, reply = await self.put(VOICE, body)
                self.assertEqual((status, reply["code"]), (400, "bad_audio"))
                self.assertNotIn("hello", reply["message"])
        self.assertEqual(self.engine.builds, 0)
        self.assertEqual((await self.health())["customVoices"], [])

    async def test_ids_are_validated_and_presets_are_protected(self):
        await self.start()
        body = speech_upload()
        self.assertEqual((await self.put("NATF2.pt", body))[0], 403)
        status, _, _ = await self.send("DELETE", "/api/voices/NATF2.pt")
        self.assertEqual(status, 403)
        for bad in ("bogus", "c-a", "c-Upper", "c-a.b", "c-ab.pt", "c-" + "x" * 42):
            with self.subTest(voice=bad):
                status, reply = await self.put(bad, body)
                self.assertEqual((status, reply["code"]), (400, "bad_request"))
        self.assertEqual(self.engine.builds, 0)

    async def test_path_tricks_cannot_reach_outside_the_custom_directory(self):
        await self.start()
        before = sorted(os.listdir(self.root.name))
        for path in ("c-..", "%2e%2e%2fc-escape", "c-..%2fescape", "..%2fc-escape"):
            with self.subTest(path=path):
                status, _, _ = await self.send("PUT", f"/api/voices/{path}", speech_upload(),
                                               headers={"Content-Type": "audio/wav"})
                self.assertIn(status, (400, 404))
        self.assertEqual(sorted(os.listdir(self.root.name)), before)
        self.assertEqual(self.engine.builds, 0)


class DeleteAndSampleTests(VoiceApiTestCase):
    async def test_delete_removes_both_files_and_then_404s(self):
        await self.start()
        await self.put(VOICE, speech_upload())
        status, _, _ = await self.send("DELETE", f"/api/voices/{VOICE}")
        self.assertEqual(status, 204)
        self.assertEqual(os.listdir(self.custom), [])
        self.assertEqual((await self.health())["customVoices"], [])
        status, _, _ = await self.send("DELETE", f"/api/voices/{VOICE}")
        self.assertEqual(status, 404)
        status, _, _ = await self.send("GET", f"/api/voices/{VOICE}.wav")
        self.assertEqual(status, 404)

    async def test_samples_exist_only_for_custom_voices(self):
        await self.start()
        for path in ("/api/voices/NATF2.wav", "/api/voices/c-nothing.wav", f"/api/voices/{VOICE}",
                     "/api/voices/c-..x.wav", "/api/voices/NATF2.pt"):
            with self.subTest(path=path):
                status, _, _ = await self.send("GET", path)
                self.assertIn(status, (404, 400))

    async def test_delete_during_a_call_is_busy(self):
        await self.start()
        await self.put(VOICE, speech_upload())
        ws = await self.call(voice_prompt="NATF2.pt")
        await next_binary(ws)
        await next_binary(ws)
        status, _, data = await self.send("DELETE", f"/api/voices/{VOICE}")
        self.assertEqual((status, json.loads(data)["code"]), (409, "busy"))
        await ws.close()
        await self.wait_not_busy()


class BusyGateTests(VoiceApiTestCase):
    async def test_a_call_in_progress_makes_a_build_busy(self):
        await self.start()
        ws = await self.call(voice_prompt="NATF2.pt")
        await next_binary(ws)  # 0x04
        await next_binary(ws)  # 0x00: live
        status, body = await self.put(VOICE, speech_upload())
        self.assertEqual((status, body["code"]), (409, "busy"))
        await ws.close()
        await self.wait_not_busy()
        status, _ = await self.put(VOICE, speech_upload())
        self.assertEqual(status, 201)

    async def test_a_build_in_progress_gives_calls_busy_and_is_visible_in_health(self):
        await self.start(engine=FakeEngine(custom_voice_dir=self.custom, build_delay_s=0.8))
        build = asyncio.ensure_future(self.put(VOICE, speech_upload()))
        await asyncio.sleep(0.25)
        self.assertIs((await self.health())["busy"], True)
        ws = await self.call()
        kind, body = await next_binary(ws)
        self.assertEqual((kind, json.loads(body)["code"]), (0x05, "busy"))
        self.assertEqual(await drain_until_closed(ws), 4409)
        status, _ = await build
        self.assertEqual(status, 201)
        ws2 = await self.call()
        kind, _ = await next_binary(ws2)
        self.assertEqual(kind, 0x04)
        await ws2.close()

    async def test_a_second_build_during_a_build_is_busy(self):
        await self.start(engine=FakeEngine(custom_voice_dir=self.custom, build_delay_s=0.6))
        first = asyncio.ensure_future(self.put("c-first", speech_upload()))
        await asyncio.sleep(0.2)
        status, body = await self.put("c-second", speech_upload())
        self.assertEqual((status, body["code"]), (409, "busy"))
        self.assertEqual((await first)[0], 201)
        self.assertEqual((await self.health())["customVoices"], ["c-first.pt"])

    async def test_the_next_call_still_runs_its_system_prompts_after_a_build(self):
        await self.start()
        resets_before = self.engine.resets
        self.assertEqual((await self.put(VOICE, speech_upload()))[0], 201)
        self.assertGreater(self.engine.resets, resets_before)  # the build reset the streaming state
        ws = await self.call(voice_prompt="NATF2.pt")
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x00)  # live: the persona prompts ran after the build
        self.assertEqual(self.engine.prompts[-1]["voice_prompt"], "NATF2.pt")
        await ws.close()
        await self.wait_not_busy()

    async def test_a_custom_voice_is_usable_only_after_it_is_built(self):
        await self.start()
        with self.assertRaises(aiohttp.WSServerHandshakeError) as ctx:
            await self.call(voice_prompt=f"{VOICE}.pt")
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual((await self.put(VOICE, speech_upload()))[0], 201)
        ws = await self.call(voice_prompt=f"{VOICE}.pt")
        kind, body = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        self.assertEqual(json.loads(body)["voice"], f"{VOICE}.pt")
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x00)
        self.assertEqual(self.engine.prompts[-1]["voice_prompt"], f"{VOICE}.pt")
        await ws.close()
        await self.wait_not_busy()

    async def test_a_failed_build_is_500_keeps_the_old_voice_and_frees_the_gate(self):
        engine = FakeEngine(custom_voice_dir=self.custom)
        await self.start(engine=engine)
        self.assertEqual((await self.put(VOICE, speech_upload()))[0], 201)
        with open(os.path.join(self.custom, f"{VOICE}.wav"), "rb") as handle:
            old_sample = handle.read()
        engine.fail_build = True
        status, body = await self.put(VOICE, speech_upload(4.5, 0.1))
        self.assertEqual((status, body), (500, {"code": "internal", "message": "the voice could not be built"}))
        self.assertIs((await self.health())["busy"], False)
        with open(os.path.join(self.custom, f"{VOICE}.wav"), "rb") as handle:
            self.assertEqual(handle.read(), old_sample)
        self.assertEqual([n for n in os.listdir(self.custom) if n.startswith(".build-")], [])
        engine.fail_build = False
        ws = await self.call(voice_prompt=f"{VOICE}.pt")
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        await ws.close()
        await self.wait_not_busy()

    async def test_logs_hold_no_audio_no_token_and_no_voice_name(self):
        logger = logging.getLogger(SESSION_LOGGER)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.setLevel, logging.CRITICAL)
        await self.start()
        upload = speech_upload()
        with self.assertLogs(SESSION_LOGGER, level="INFO") as captured:
            self.assertEqual((await self.put(VOICE, upload))[0], 201)
        joined = "\n".join(captured.output)
        self.assertIn("custom voice built", joined)
        for secret in (TOKEN, VOICE, upload[44:64].hex(), "RIFF"):
            self.assertNotIn(secret, joined)


if __name__ == "__main__":
    unittest.main()
