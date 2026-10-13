# SPDX-License-Identifier: MIT
"""Integration tests for moshi.sim_session over real sockets (aiohttp TestServer, fake engine)."""

import asyncio
import json
import logging
import re
import time
import unittest
from urllib.parse import urlencode

import aiohttp
import numpy as np
from aiohttp import WSMsgType, web
from aiohttp.test_utils import unused_port

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
from fake_engine import FakeEngine
from moshi.sim_session import SessionConfig, create_app

# asyncio debug mode (used by IsolatedAsyncioTestCase) logs slow callbacks at WARNING; keep output clean.
logging.getLogger("asyncio").setLevel(logging.ERROR)

TOKEN = "test-token-0123456789abcdef"
FRAME_BYTES = 1920 * 2
SILENCE = bytes(FRAME_BYTES)
SESSION_LOGGER = "moshi.sim_session"


def ints_to_bytes(samples: np.ndarray) -> bytes:
    return samples.astype("<i2").tobytes()


async def next_binary(ws, timeout=5.0):
    msg = await asyncio.wait_for(ws.receive(), timeout)
    if msg.type is not WSMsgType.BINARY:
        raise AssertionError(f"expected a binary message, got {msg.type}")
    return msg.data[0], msg.data[1:]


async def expect_kind(ws, kind, timeout=5.0):
    """Return the body of the first message of this kind; other kinds are skipped."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"timed out waiting for kind {kind:#04x}")
        msg = await asyncio.wait_for(ws.receive(), remaining)
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
            raise AssertionError(f"socket closed while waiting for kind {kind:#04x}")
        if msg.type is WSMsgType.BINARY and msg.data[0] == kind:
            return msg.data[1:]


async def read_error(ws, timeout=5.0):
    return json.loads(await expect_kind(ws, 0x05, timeout))


async def drain_until_closed(ws, timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("timed out waiting for the server to close")
        msg = await asyncio.wait_for(ws.receive(), remaining)
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
            return ws.close_code


class SimSessionTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        logger = logging.getLogger(SESSION_LOGGER)
        previous = logger.level
        logger.setLevel(logging.CRITICAL)  # keep test output quiet unless a test asks for logs
        self.addCleanup(logger.setLevel, previous)
        self.http = aiohttp.ClientSession()
        self.addAsyncCleanup(self.http.close)
        self.engine = None
        self.cfg = None

    async def start(self, engine=None, handler_cancellation=False, **config):
        """Serve the app with the runner settings web.run_app uses in production.

        handler_cancellation=False is production's default; True covers aiohttp cancelling the
        handler when the client drops (aiohttp's TestServer enables it, so it is tested on purpose).
        """
        self.engine = engine or FakeEngine()
        config.setdefault("auth_token", TOKEN)
        self.cfg = SessionConfig(**config)
        runner = web.AppRunner(create_app(self.engine, self.cfg),
                               handler_cancellation=handler_cancellation, access_log=None)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        port = unused_port()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        self.base = f"127.0.0.1:{port}"

    def ws_url(self, **params):
        query = urlencode(params)
        return f"ws://{self.base}/api/chat" + (f"?{query}" if query else "")

    def auth(self):
        return {"Authorization": f"Bearer {TOKEN}"}

    async def connect(self, headers=None, **params):
        return await self.http.ws_connect(self.ws_url(**params),
                                          headers=self.auth() if headers is None else headers)

    async def start_call(self, codec="pcm16", **params):
        """Connect, read 0x04 accepted and 0x00 handshake. Returns (ws, accepted)."""
        ws = await self.connect(codec=codec, **params)
        kind, body = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        accepted = json.loads(body)
        kind, body = await next_binary(ws)
        self.assertEqual((kind, body), (0x00, b""))
        return ws, accepted

    async def health(self):
        async with self.http.get(f"http://{self.base}/api/health", headers=self.auth()) as resp:
            self.assertEqual(resp.status, 200)
            return await resp.json()

    async def wait_not_busy(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while (await self.health())["busy"]:
            if time.monotonic() > deadline:
                raise AssertionError("the server still reports busy")
            await asyncio.sleep(0.02)


class AuthAndValidationTests(SimSessionTestCase):
    async def test_health_requires_the_bearer_token(self):
        await self.start()
        url = f"http://{self.base}/api/health"
        for headers in ({}, {"Authorization": "Bearer wrong-token-value-0000"}, {"Authorization": f"Basic {TOKEN}"}):
            with self.subTest(headers=sorted(headers)):
                async with self.http.get(url, headers=headers) as resp:
                    self.assertEqual(resp.status, 401)
                    self.assertEqual(resp.headers.get("WWW-Authenticate"), "Bearer")
        async with self.http.get(url, headers=self.auth()) as resp:
            self.assertEqual(resp.status, 200)
            body = await resp.json()
        self.assertEqual(set(body), {"ok", "busy", "model", "fast", "voices", "uptimeS"})
        self.assertIs(body["ok"], True)
        self.assertIs(body["busy"], False)
        self.assertEqual(body["model"], "personaplex-7b-v1")
        self.assertIs(body["fast"], False)
        self.assertEqual(body["voices"], ["NATF0.pt", "NATF2.pt"])
        self.assertIsInstance(body["uptimeS"], int)

    async def test_unknown_api_paths_need_the_token_too(self):
        await self.start()
        async with self.http.get(f"http://{self.base}/api/nothing-here") as resp:
            self.assertEqual(resp.status, 401)

    async def test_chat_is_refused_before_the_upgrade_without_a_valid_token(self):
        await self.start()
        for headers in ({}, {"Authorization": "Bearer wrong-token-value-0000"}):
            with self.subTest(headers=sorted(headers)):
                with self.assertRaises(aiohttp.WSServerHandshakeError) as ctx:
                    await self.connect(headers=headers, codec="pcm16")
                self.assertEqual(ctx.exception.status, 401)

    async def test_bad_parameters_are_400_before_the_upgrade(self):
        await self.start()
        cases = [
            {"voice_prompt": "../NATF2.pt"},
            {"voice_prompt": "NATF9.pt"},
            {"voice_prompt": "NATF2.wav"},
            {"voice_prompt": "NATF2.pt\n"},
            {"text_prompt": "x" * 1201},
            {"session": "has space"},
            {"session": "a" * 65},
            {"codec": "mp3"},
            {"seed": "abc"},
            {"seed": str(2**32)},
            {"seed": "-2"},
        ]
        for params in cases:
            with self.subTest(params=sorted(params)):
                with self.assertRaises(aiohttp.WSServerHandshakeError) as ctx:
                    await self.connect(**params)
                self.assertEqual(ctx.exception.status, 400)

    async def test_bad_request_body_is_json_and_never_echoes_input(self):
        await self.start()
        secret_like = "PERSONA-SENTINEL-" + "x" * 1200
        url = f"http://{self.base}/api/chat?" + urlencode({"text_prompt": secret_like})
        async with self.http.get(url, headers=self.auth()) as resp:
            self.assertEqual(resp.status, 400)
            body = await resp.json()
        self.assertEqual(body["code"], "bad_request")
        self.assertNotIn("PERSONA-SENTINEL", json.dumps(body))


class HandshakeAndAudioTests(SimSessionTestCase):
    async def test_accepted_then_handshake(self):
        await self.start()
        ws, accepted = await self.start_call(session="call-1", voice_prompt="NATF0.pt")
        self.assertEqual(accepted, {
            "type": "accepted",
            "session": "call-1",
            "sampleRate": 24000,
            "frameMs": 80,
            "codec": "pcm16",
            "voice": "NATF0.pt",
            "model": "personaplex-7b-v1",
            "fast": False,
        })
        self.assertEqual(self.engine.prompts[-1]["voice_prompt"], "NATF0.pt")
        await ws.close()

    async def test_pcm16_frames_come_back_as_frames(self):
        await self.start()
        ws, _ = await self.start_call()
        samples = np.random.default_rng(3).integers(-32768, 32768, size=1920 * 3, dtype=np.int32)
        for index in range(3):
            await ws.send_bytes(b"\x11" + ints_to_bytes(samples[index * 1920:(index + 1) * 1920]))
        audio, texts = [], []
        while len(audio) < 3 or len(texts) < 3:
            kind, body = await next_binary(ws)
            if kind == 0x11:
                self.assertEqual(len(body), 3840)
                audio.append(np.frombuffer(body, "<i2"))
            elif kind == 0x02:
                texts.append(body.decode("utf-8"))
        self.assertTrue(np.array_equal(np.concatenate(audio), samples.astype("<i2")))
        self.assertEqual(texts, ["a", "a", "a"])
        self.assertEqual(self.engine.steps, 3)
        await ws.close()

    async def test_odd_sized_client_chunks_are_reframed(self):
        await self.start()
        ws, _ = await self.start_call()
        samples = np.random.default_rng(5).integers(-32768, 32768, size=5760, dtype=np.int32)
        for start in range(0, samples.size, 1000):
            await ws.send_bytes(b"\x11" + ints_to_bytes(samples[start:start + 1000]))
        audio = []
        while len(audio) < 3:
            kind, body = await next_binary(ws)
            if kind == 0x11:
                audio.append(np.frombuffer(body, "<i2"))
        self.assertTrue(np.array_equal(np.concatenate(audio), samples.astype("<i2")))
        self.assertEqual(self.engine.steps, 3)
        await ws.close()

    async def test_audio_before_the_handshake_is_discarded(self):
        await self.start(engine=FakeEngine(prompt_delay_s=0.3))
        ws = await self.connect(codec="pcm16")
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        for _ in range(4):
            await ws.send_bytes(b"\x11" + SILENCE)
        kind, body = await next_binary(ws)
        self.assertEqual((kind, body), (0x00, b""))
        await asyncio.sleep(0.1)
        self.assertEqual(self.engine.steps, 0)

        for _ in range(2):
            await ws.send_bytes(b"\x11" + SILENCE)
        received = 0
        while received < 2:
            kind, _ = await next_binary(ws)
            received += kind == 0x11
        self.assertEqual(self.engine.steps, 2)
        await ws.close()

    async def test_session_id_is_generated_when_absent(self):
        await self.start()
        ws, accepted = await self.start_call()
        self.assertRegex(accepted["session"], re.compile(r"^[0-9a-f]{16}$"))
        await ws.close()

    async def test_defaults_and_explicit_prompt_values_reach_the_engine(self):
        engine = FakeEngine()
        await self.start(engine=engine, default_text_prompt="Be brief.")
        ws, _ = await self.start_call()
        self.assertEqual(engine.prompts[-1], {"voice_prompt": "NATF2.pt", "text_prompt": "Be brief.", "seed": None})
        await ws.close()
        await self.wait_not_busy()

        ws, _ = await self.start_call(voice_prompt="NATF0.pt", text_prompt="Say hi.", seed="42")
        self.assertEqual(engine.prompts[-1], {"voice_prompt": "NATF0.pt", "text_prompt": "Say hi.", "seed": 42})
        await ws.close()
        await self.wait_not_busy()

        ws, _ = await self.start_call(seed="-1")
        self.assertIsNone(engine.prompts[-1]["seed"])
        await ws.close()


class CallLifecycleTests(SimSessionTestCase):
    async def test_second_client_gets_busy_and_4409_then_the_slot_frees(self):
        await self.start()
        ws1, _ = await self.start_call()
        self.assertIs((await self.health())["busy"], True)

        ws2 = await self.connect(codec="pcm16")
        kind, body = await next_binary(ws2)
        self.assertEqual(kind, 0x05)
        self.assertEqual(json.loads(body)["code"], "busy")
        self.assertEqual(await drain_until_closed(ws2), 4409)

        await ws1.close()
        await self.wait_not_busy()
        ws3, accepted = await self.start_call()
        self.assertEqual(accepted["type"], "accepted")
        await ws3.close()

    async def test_idle_timeout_ends_the_call_with_4408(self):
        await self.start(idle_timeout_s=0.4, stats_interval_s=0.1)
        ws, _ = await self.start_call()
        error = await read_error(ws, timeout=5.0)
        self.assertEqual(error["code"], "timeout")
        self.assertEqual(await drain_until_closed(ws), 4408)

    async def test_max_session_ends_the_call_with_4408(self):
        await self.start(max_session_s=0.5, idle_timeout_s=30.0)
        ws, _ = await self.start_call()
        error = await read_error(ws, timeout=5.0)
        self.assertEqual(error["code"], "timeout")
        self.assertIn("max session", error["message"])
        self.assertEqual(await drain_until_closed(ws), 4408)

    async def test_engine_failure_closes_with_internal_4500(self):
        await self.start(engine=FakeEngine(fail_on_step=True))
        ws, _ = await self.start_call()
        await ws.send_bytes(b"\x11" + SILENCE)
        error = await read_error(ws)
        self.assertEqual(error, {"code": "internal", "message": "internal error"})
        self.assertEqual(await drain_until_closed(ws), 4500)

    async def test_client_leaving_during_prompts_frees_the_model(self):
        engine = FakeEngine(prompt_delay_s=5.0)
        await self.start(engine=engine)
        ws = await self.connect(codec="pcm16")
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        await ws.close()
        await self.wait_not_busy()
        self.assertEqual(engine.aborted_prompts, 1)

        engine.prompt_delay_s = 0.0
        ws2, _ = await self.start_call()
        await ws2.close()

    async def test_handler_cancellation_also_frees_the_model(self):
        # aiohttp's TestServer cancels handlers on disconnect; the call must still release busy and
        # leave no recv task behind, so the next call is accepted.
        engine = FakeEngine(prompt_delay_s=5.0)
        await self.start(engine=engine, handler_cancellation=True)
        ws = await self.connect(codec="pcm16")
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        await ws.close()
        await self.wait_not_busy()

        engine.prompt_delay_s = 0.0
        ws2, _ = await self.start_call()
        await ws2.close()

    async def test_stats_message_has_the_contract_keys(self):
        await self.start(stats_interval_s=0.1)
        ws, _ = await self.start_call()
        stats = json.loads(await expect_kind(ws, 0x06, timeout=3.0))
        self.assertEqual(stats["type"], "stats")
        self.assertIn("msPerFrameP50", stats)
        self.assertEqual(stats["backlogFrames"], 0)
        self.assertEqual(stats["droppedFrames"], 0)
        await ws.close()

    async def test_backlog_is_bounded_and_drops_are_counted(self):
        await self.start(engine=FakeEngine(step_delay_s=0.03), max_backlog_frames=2,
                         stats_interval_s=0.1, idle_timeout_s=30.0)
        ws, _ = await self.start_call()
        for _ in range(30):
            await ws.send_bytes(b"\x11" + SILENCE)
        stats = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            kind, body = await next_binary(ws, timeout=3.0)
            if kind == 0x06:
                candidate = json.loads(body)
                if candidate["droppedFrames"] > 0:
                    stats = candidate
                    break
        self.assertIsNotNone(stats, "no stats message reported dropped frames")
        self.assertLessEqual(stats["backlogFrames"], 2)
        self.assertLess(self.engine.steps, 30)
        await ws.close()

    async def test_logs_never_contain_prompt_text_transcripts_or_the_token(self):
        logging.getLogger(SESSION_LOGGER).setLevel(logging.INFO)
        await self.start(engine=FakeEngine(text_piece="TRANSCRIPT-SENTINEL"))
        with self.assertLogs(level="INFO") as captured:
            ws, _ = await self.start_call(text_prompt="PERSONA-SENTINEL speaks")
            await ws.send_bytes(b"\x11" + SILENCE)
            await expect_kind(ws, 0x02)  # the text piece really was sent
            await ws.close()
            await self.wait_not_busy()
        joined = "\n".join(captured.output)
        self.assertIn("session", joined)
        for secret in ("PERSONA-SENTINEL", "TRANSCRIPT-SENTINEL", TOKEN):
            self.assertNotIn(secret, joined)


class OpusRoutingTests(SimSessionTestCase):
    async def test_opus_sessions_use_kind_0x01_and_ignore_the_other_kind(self):
        await self.start()
        ws = await self.connect(codec="opus")
        kind, body = await next_binary(ws)
        self.assertEqual(kind, 0x04)
        self.assertEqual(json.loads(body)["codec"], "opus")
        kind, _ = await next_binary(ws)
        self.assertEqual(kind, 0x00)

        await ws.send_bytes(b"\x11" + SILENCE)  # wrong kind for this codec: ignored
        await ws.send_bytes(b"\x01" + b"opus-frame")
        self.assertEqual(await expect_kind(ws, 0x01), b"OPUS")
        self.assertEqual(self.engine.steps, 1)
        await ws.close()

    async def test_undecodable_audio_closes_with_bad_request_4400(self):
        await self.start()
        ws = await self.connect(codec="opus")
        await next_binary(ws)
        await next_binary(ws)
        await ws.send_bytes(b"\x01bad")
        error = await read_error(ws)
        self.assertEqual(error["code"], "bad_request")
        self.assertEqual(await drain_until_closed(ws), 4400)


class ConfigValidationTests(unittest.TestCase):
    def test_bad_limits_are_refused(self):
        with self.assertRaises(ValueError):
            SessionConfig(max_backlog_frames=0)
        with self.assertRaises(ValueError):
            SessionConfig(idle_timeout_s=0)
        with self.assertRaises(ValueError):
            SessionConfig(max_session_s=-1)
        with self.assertRaises(ValueError):
            SessionConfig(stats_interval_s=0)

    def test_token_is_not_shown_in_repr(self):
        self.assertNotIn(TOKEN, repr(SessionConfig(auth_token=TOKEN)))

    def test_empty_or_short_tokens_are_refused_not_silently_unauthenticated(self):
        with self.assertRaises(ValueError):
            SessionConfig(auth_token="")
        with self.assertRaises(ValueError):
            SessionConfig(auth_token="short")
        self.assertIsNone(SessionConfig(auth_token=None).auth_token)

    def test_outbox_overflow_ends_the_call_instead_of_growing_forever(self):
        from moshi.sim_session import _Call
        from moshi.sim_protocol import ChatParams

        params = ChatParams(codec="pcm16", voice_prompt="NATF2.pt", text_prompt="x", seed=None, session="s1")
        call = _Call(ws=None, engine=None, config=SessionConfig(outbox_limit=8), params=params)
        for _ in range(8):
            call._enqueue(b"\x11")
        self.assertIsNone(call.end)
        call._enqueue(b"\x11")  # one more than the limit
        self.assertEqual(call.end[0], "timeout")
        self.assertEqual(call.end[2], 4408)
        self.assertTrue(call.done.is_set())

    def test_bearer_check_fails_closed_for_empty_expectation(self):
        from moshi.sim_protocol import bearer_matches
        self.assertFalse(bearer_matches("Bearer ", b""))
        self.assertFalse(bearer_matches("Bearer anything-at-all-here", b""))


if __name__ == "__main__":
    unittest.main()
