# SPDX-License-Identifier: MIT
"""Fake engine for the Sim session tests: no torch, no GPU.

Echoes each input frame back as output audio and emits a fixed text piece per frame, and records
the system-prompt arguments so tests can check defaults and validation.
"""

import asyncio
import time

import numpy as np

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)
from moshi.sim_protocol import Pcm16Codec


class FakeOpusCodec:
    """Stands in for the sphn-backed codec: wire kind 0x01, payloads are opaque."""

    in_kind = 0x01
    out_kind = 0x01

    def decode(self, payload: bytes) -> np.ndarray:
        if payload == b"bad":
            raise ValueError("undecodable payload")
        return np.zeros(1920, dtype=np.float32) if payload else np.zeros(0, dtype=np.float32)

    def encode(self, pcm: np.ndarray) -> bytes:
        return b"OPUS"


class FakeEngine:
    def __init__(self, voices=("NATF0.pt", "NATF2.pt"), prompt_delay_s=0.0, step_delay_s=0.0,
                 fail_on_step=False, text_piece="a"):
        self._voices = list(voices)
        self.prompt_delay_s = prompt_delay_s
        self.step_delay_s = step_delay_s
        self.fail_on_step = fail_on_step
        self.text_piece = text_piece
        self.prompts: list[dict] = []
        self.aborted_prompts = 0
        self.steps = 0
        self.resets = 0

    def voices(self) -> list[str]:
        return list(self._voices)

    async def run_system_prompts(self, *, voice_prompt, text_prompt, seed, is_alive) -> None:
        self.prompts.append({"voice_prompt": voice_prompt, "text_prompt": text_prompt, "seed": seed})
        deadline = time.monotonic() + self.prompt_delay_s
        while time.monotonic() < deadline:
            if not await is_alive():
                self.aborted_prompts += 1
                return
            await asyncio.sleep(0.01)

    def reset(self) -> None:
        self.resets += 1

    def step(self, pcm: np.ndarray) -> list:
        if self.fail_on_step:
            raise RuntimeError("fake engine failure")
        if self.step_delay_s:
            time.sleep(self.step_delay_s)  # blocking, like a GPU step
        self.steps += 1
        return [(pcm.copy(), self.text_piece or None)]

    def new_codec(self, codec: str):
        if codec == "pcm16":
            return Pcm16Codec()
        if codec == "opus":
            return FakeOpusCodec()
        raise ValueError(codec)
