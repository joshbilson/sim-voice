# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# HTTP + WebSocket session layer for the Sim voice server: bearer auth, request validation,
# one call at a time (busy), bounded input latency, idle and session limits, stats.
# Torch-free by design: no torch, sphn or sentencepiece imports. The engine (server.ServerState)
# owns the model and the Opus codec. Protocol summary: SIM.md.
#
# Privacy: this module never logs text prompts, transcripts, audio or the token. It logs the
# session id, timings, frame counts and close codes only.

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Protocol, Sequence

import numpy as np
from aiohttp import WSMsgType, web

from .sim_protocol import (
    CLOSE_NORMAL,
    DEFAULT_TEXT_PROMPT,
    DEFAULT_VOICE,
    ERROR_CLOSE_CODES,
    FRAME_MS,
    FRAME_SAMPLES,
    KIND_ACCEPTED,
    KIND_HANDSHAKE,
    KIND_STATS,
    KIND_TEXT,
    MODEL_NAME,
    SAMPLE_RATE,
    BadRequest,
    MIN_TOKEN_CHARS,
    FrameBuffer,
    LatencyWindow,
    bearer_matches,
    error_message,
    json_message,
    parse_chat_query,
)

logger = logging.getLogger(__name__)


class AudioCodec(Protocol):
    """Converts between wire payloads and float32 PCM at 24 kHz (see sim_protocol.Pcm16Codec)."""

    in_kind: int
    out_kind: int

    def decode(self, payload: bytes) -> np.ndarray: ...

    def encode(self, pcm: np.ndarray) -> bytes: ...


class SimEngine(Protocol):
    """What the session layer needs from the model. server.ServerState implements it."""

    def voices(self) -> Sequence[str]:
        """Voice prompt file names (e.g. NATF2.pt) available on this server."""

    async def run_system_prompts(
        self,
        *,
        voice_prompt: str,
        text_prompt: str,
        seed: Optional[int],
        is_alive: Callable[[], Awaitable[bool]],
    ) -> None:
        """Condition the model on the voice and persona; returns early when is_alive() is False."""

    def reset(self) -> None:
        """Clear streaming state so the next call starts clean."""

    def step(self, pcm: np.ndarray) -> list:
        """Feed one 1920-sample float32 frame. Returns [(pcm_out | None, text_piece | None), ...]."""

    def new_codec(self, codec: str) -> AudioCodec:
        """A fresh codec object for one call ('pcm16' or 'opus')."""


@dataclass
class SessionConfig:
    fast: bool = False
    default_voice: str = DEFAULT_VOICE
    default_text_prompt: str = DEFAULT_TEXT_PROMPT
    auth_token: Optional[str] = field(default=None, repr=False)
    max_backlog_frames: int = 6
    idle_timeout_s: float = 15.0
    max_session_s: float = 1800.0
    stats_interval_s: float = 2.0
    outbox_limit: int = 256

    def __post_init__(self) -> None:
        if self.max_backlog_frames < 1:
            raise ValueError("max_backlog_frames must be at least 1")
        for name in ("idle_timeout_s", "max_session_s", "stats_interval_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.outbox_limit < 8:
            raise ValueError("outbox_limit must be at least 8")
        if self.auth_token is not None and len(self.auth_token) < MIN_TOKEN_CHARS:
            raise ValueError(f"auth_token must be at least {MIN_TOKEN_CHARS} characters")


def _auth_middleware(token: Optional[str]):
    # None means loopback-only with auth off; an empty token is refused by SessionConfig.
    expected = token.encode("utf-8") if token is not None else None

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        # Every /api/* request is checked before any upgrade or handler runs.
        if expected is not None and request.path.startswith("/api/"):
            if not bearer_matches(request.headers.get("Authorization"), expected):
                return web.json_response(
                    {"code": "unauthorized", "message": "missing or invalid bearer token"},
                    status=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await handler(request)

    return auth_middleware


def create_app(engine: SimEngine, config: SessionConfig) -> web.Application:
    server = SimServer(engine, config)
    app = web.Application(middlewares=[_auth_middleware(config.auth_token)])
    app.router.add_get("/api/chat", server.handle_chat)
    app.router.add_get("/api/health", server.handle_health)
    return app


class SimServer:
    def __init__(self, engine: SimEngine, config: SessionConfig) -> None:
        self.engine = engine
        self.config = config
        self.busy = False
        self._started = time.monotonic()

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "ok": True,
            "busy": self.busy,
            "model": MODEL_NAME,
            "fast": self.config.fast,
            "voices": list(self.engine.voices()),
            "uptimeS": int(time.monotonic() - self._started),
        })

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        try:
            params = parse_chat_query(
                request.query,
                voices=self.engine.voices(),
                default_voice=self.config.default_voice,
                default_text_prompt=self.config.default_text_prompt,
            )
        except BadRequest as exc:
            logger.info("chat request rejected: %s", exc)
            return web.json_response({"code": "bad_request", "message": str(exc)}, status=400)

        ws = web.WebSocketResponse(timeout=5.0)
        await ws.prepare(request)
        if self.busy:
            logger.info("session %s rejected: busy", params.session)
            await _send_error(ws, "busy", "a call is already in progress")
            await ws.close(code=ERROR_CLOSE_CODES["busy"], message=b"")
            return ws

        # No await between the check above and this assignment: one call at a time, no queue.
        self.busy = True
        try:
            await _Call(ws, self.engine, self.config, params).run()
        finally:
            self.busy = False
            try:
                self.engine.reset()
            except Exception as exc:
                logger.error("session %s: engine reset failed (%s)", params.session, type(exc).__name__)
        return ws


async def _send_error(ws: web.WebSocketResponse, code: str, message: str) -> None:
    if ws.closed:
        return
    try:
        await ws.send_bytes(error_message(code, message))
    except Exception:
        pass  # the peer is already gone


class _Call:
    """One call: accept, handshake, then receive/process/send/watch until someone finishes it."""

    def __init__(self, ws: web.WebSocketResponse, engine: SimEngine, config: SessionConfig, params) -> None:
        self.ws = ws
        self.engine = engine
        self.cfg = config
        self.params = params
        self.sid: str = params.session
        self.codec: Optional[AudioCodec] = None
        self.inbox = FrameBuffer(FRAME_SAMPLES, config.max_backlog_frames)
        self.latency = LatencyWindow()
        self.outbox: asyncio.Queue = asyncio.Queue(maxsize=config.outbox_limit)
        self.frames_ready = asyncio.Event()
        self.done = asyncio.Event()
        # (error code or None for a normal end, message, close code)
        self.end: Optional[tuple] = None
        self.live = False
        self.client_gone = False
        self.frames_in = 0
        self.frames_out = 0
        self.started = time.monotonic()
        self.handshake_at = self.started
        self.last_audio_at = self.started
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        try:
            await self._start()
            await self.done.wait()
        finally:
            # Also runs when aiohttp cancels the handler (handler_cancellation): no task outlives the call.
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._close()

    async def _start(self) -> None:
        try:
            self.codec = self.engine.new_codec(self.params.codec)
            accepted = {
                "type": "accepted",
                "session": self.sid,
                "sampleRate": SAMPLE_RATE,
                "frameMs": FRAME_MS,
                "codec": self.params.codec,
                "voice": self.params.voice_prompt,
                "model": MODEL_NAME,
                "fast": self.cfg.fast,
            }
            if not await self._send(json_message(KIND_ACCEPTED, accepted)):
                return
            logger.info("session %s accepted codec=%s voice=%s", self.sid, self.params.codec, self.params.voice_prompt)

            self._tasks.append(asyncio.create_task(self._recv_loop()))
            await self._run_system_prompts()
            if self.done.is_set():
                return

            if not await self._send(bytes([KIND_HANDSHAKE])):
                return
            self.live = True
            self.handshake_at = self.last_audio_at = time.monotonic()
            logger.info("session %s live after %.0f ms", self.sid, (self.handshake_at - self.started) * 1000.0)
            self._tasks.append(asyncio.create_task(self._process_loop()))
            self._tasks.append(asyncio.create_task(self._send_loop()))
            self._tasks.append(asyncio.create_task(self._watch_loop()))
        except Exception as exc:
            self._internal("start", exc)

    async def _run_system_prompts(self) -> None:
        try:
            await self.engine.run_system_prompts(
                voice_prompt=self.params.voice_prompt,
                text_prompt=self.params.text_prompt,
                seed=self.params.seed,
                is_alive=self._alive,
            )
        except Exception as exc:
            self._internal("system prompts", exc)
            return
        if not await self._alive():
            self._finish(None)

    async def _alive(self) -> bool:
        return not self.client_gone and not self.done.is_set() and not self.ws.closed

    async def _recv_loop(self) -> None:
        try:
            async for msg in self.ws:
                if msg.type is WSMsgType.BINARY:
                    if msg.data and self.live and not self._on_audio(msg.data):
                        break
                elif msg.type is WSMsgType.ERROR:
                    self._finish("bad_request", "malformed websocket message")
                    break
                # Other message types are ignored; audio before the handshake is discarded.
        except Exception:
            pass  # socket error: treated as the client leaving
        self.client_gone = True
        self.frames_ready.set()
        self._finish(None)

    def _on_audio(self, data: bytes) -> bool:
        assert self.codec is not None
        if data[0] != self.codec.in_kind:
            return True  # wrong kind for this codec: ignored
        self.last_audio_at = time.monotonic()
        try:
            pcm = self.codec.decode(data[1:])
        except Exception:
            self._finish("bad_request", "invalid audio payload")
            return False
        self.inbox.push(pcm)
        self.frames_ready.set()
        return True

    async def _process_loop(self) -> None:
        assert self.codec is not None
        try:
            while True:
                frame = self.inbox.pop()
                if frame is None:
                    self.frames_ready.clear()
                    await self.frames_ready.wait()
                    continue
                started = time.perf_counter()
                outputs = self.engine.step(frame)
                self.latency.add((time.perf_counter() - started) * 1000.0)
                self.frames_in += 1
                for pcm, text in outputs:
                    if pcm is not None:
                        self.frames_out += 1
                        body = self.codec.encode(pcm)
                        if body:
                            self._enqueue(bytes([self.codec.out_kind]) + body)
                    if text:
                        self._enqueue(bytes([KIND_TEXT]) + text.encode("utf-8"))
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._internal("step", exc)

    async def _send_loop(self) -> None:
        while True:
            message = await self.outbox.get()
            if not await self._send(message):
                return

    async def _watch_loop(self) -> None:
        tick = max(0.01, min(0.25, self.cfg.stats_interval_s, self.cfg.idle_timeout_s / 2))
        next_stats = time.monotonic() + self.cfg.stats_interval_s
        while True:
            await asyncio.sleep(tick)
            now = time.monotonic()
            if now - self.last_audio_at > self.cfg.idle_timeout_s:
                self._finish("timeout", "no audio from the client")
                return
            if now - self.handshake_at > self.cfg.max_session_s:
                self._finish("timeout", "max session length reached")
                return
            if now >= next_stats:
                next_stats = now + self.cfg.stats_interval_s
                self._enqueue(json_message(KIND_STATS, {
                    "type": "stats",
                    "msPerFrameP50": self.latency.p50(),
                    "backlogFrames": len(self.inbox),
                    "droppedFrames": self.inbox.dropped,
                }))

    def _enqueue(self, message: bytes) -> None:
        try:
            self.outbox.put_nowait(message)
        except asyncio.QueueFull:
            self._finish("timeout", "client is not reading the call audio")

    async def _send(self, data: bytes) -> bool:
        try:
            await self.ws.send_bytes(data)
            return True
        except Exception:
            self._finish(None)  # the peer went away
            return False

    def _finish(self, code: Optional[str], message: str = "") -> None:
        if self.end is None:
            close_code = CLOSE_NORMAL if code is None else ERROR_CLOSE_CODES[code]
            self.end = (code, message, close_code)
        self.done.set()

    def _internal(self, where: str, exc: BaseException) -> None:
        # Exception type and location only: messages can echo request data.
        location = ""
        if exc.__traceback__ is not None:
            frames = traceback.extract_tb(exc.__traceback__)
            if frames:
                location = f" at {os.path.basename(frames[-1].filename)}:{frames[-1].lineno}"
        logger.error("session %s failed in %s: %s%s", self.sid, where, type(exc).__name__, location)
        self._finish("internal", "internal error")

    async def _close(self) -> None:
        code, message, close_code = self.end or (None, "", CLOSE_NORMAL)
        if code is not None:
            await _send_error(self.ws, code, message)
        if not self.ws.closed:
            try:
                await self.ws.close(code=close_code, message=b"")
            except Exception:
                pass
        logger.info(
            "session %s closed code=%d%s duration=%.1fs frames_in=%d frames_out=%d dropped=%d p50_ms=%s",
            self.sid,
            close_code,
            f" reason={code}" if code else "",
            time.monotonic() - self.started,
            self.frames_in,
            self.frames_out,
            self.inbox.dropped,
            self.latency.p50(),
        )
