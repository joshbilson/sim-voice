# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Sim fork (joshbilson/sim-voice): this module is the engine adapter (ServerState) and the CLI.
# The HTTP/WebSocket protocol lives in sim_session.py and sim_protocol.py, which do not import
# torch and are tested without a GPU. Protocol and flags: SIM.md.

import argparse
import os
import random
import ssl
import tarfile
from pathlib import Path
from typing import Optional
import typing as tp

import numpy as np
import sentencepiece
import sphn
import torch
from aiohttp import web

from .models import loaders, MimiModel, LMModel, LMGen
from .sim_cli import build_arg_parser, validate_serve_args
from .sim_protocol import (
    FRAME_SAMPLES,
    KIND_OPUS,
    MODEL_NAME,
    VOICE_RE,
    Pcm16Codec,
)
from .sim_session import SessionConfig, create_app
from .utils.logging import setup_logger
from .w8a16_checkpoint import load_lm_w8a16_checkpoint


logger = setup_logger(__name__)
setup_logger("moshi.sim_session")


def torch_auto_device(requested: Optional[str] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def list_voice_files(directory: str) -> list[str]:
    """Voice prompt files (NAT*.pt, VAR*.pt, ...) in a directory, sorted."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        name for name in names
        if VOICE_RE.fullmatch(name) and os.path.isfile(os.path.join(directory, name))
    )


class OpusCodec:
    """codec=opus: Ogg Opus in and out through sphn, as in upstream's serve path."""

    in_kind = KIND_OPUS
    out_kind = KIND_OPUS

    def __init__(self, sample_rate: int) -> None:
        self._reader = sphn.OpusStreamReader(sample_rate)
        self._writer = sphn.OpusStreamWriter(sample_rate)

    def decode(self, payload: bytes) -> np.ndarray:
        self._reader.append_bytes(payload)
        return self._reader.read_pcm()

    def encode(self, pcm: np.ndarray) -> bytes:
        self._writer.append_pcm(pcm)
        return self._writer.read_bytes()


class ServerState:
    """Engine for sim_session: Mimi codecs, the LM and its streaming state. One call at a time."""

    def __init__(self, mimi: MimiModel, other_mimi: tp.Optional[MimiModel],
                 text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False, fp8: bool = False,
                 pinned_io: bool = False, dep_q_exit: int = 0):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self._fp8 = fp8
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        if self.frame_size != FRAME_SAMPLES:
            raise RuntimeError(f"model frame is {self.frame_size} samples; the Sim protocol needs {FRAME_SAMPLES}")
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
                            depformer_early_exit=dep_q_exit,
        )
        self._input_dtype = next(self.mimi.parameters()).dtype
        # Optional pinned CPU buffer for non-blocking DtoH audio transfer (mimi decode is 1 x 1 x 1920).
        self._pinned_pcm = (torch.empty(FRAME_SAMPLES, dtype=torch.float32, pin_memory=True)
                            if pinned_io else None)
        self.mimi.streaming_forever(1)
        if self.other_mimi is not None:
            self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        # dtype follows mimi (fp32 stock; fp16 only under --mimi-fp16)
        warmup_dtype = next(self.mimi.parameters()).dtype
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=warmup_dtype, device=self.device)
            codes = self.mimi.encode(chunk)
            if self.other_mimi is not None:
                _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                if self.other_mimi is not None:
                    _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    # ---- engine protocol (see sim_session.SimEngine) ----

    def voices(self) -> list[str]:
        return list_voice_files(self.voice_prompt_dir) if self.voice_prompt_dir else []

    async def run_system_prompts(self, *, voice_prompt: str, text_prompt: str,
                                 seed: Optional[int], is_alive) -> None:
        # The session layer has validated voice_prompt (regex + listed in the voice dir); checked again here.
        if os.path.basename(voice_prompt) != voice_prompt or not VOICE_RE.fullmatch(voice_prompt):
            raise ValueError("voice prompt must be a plain file name")
        voice_path = os.path.join(self.voice_prompt_dir, voice_prompt)
        if self.lm_gen.voice_prompt != voice_path:
            if voice_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_path)
            else:
                self.lm_gen.load_voice_prompt(voice_path)
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
        if seed is not None:
            seed_all(seed)
        self.reset()
        # Reuses mimi for the voice prompt, then resets it before the conversation starts.
        await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
        self.mimi.reset_streaming()

    def reset(self) -> None:
        self.mimi.reset_streaming()
        if self.other_mimi is not None:
            self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def step(self, pcm: np.ndarray) -> list:
        """One 1920-sample float32 frame in; [(pcm_out, text_piece), ...] out (both may be None)."""
        chunk = torch.from_numpy(pcm).to(device=self.device, dtype=self._input_dtype)[None, None]
        codes = self.mimi.encode(chunk)
        if self.other_mimi is not None:
            # Output always discarded; skippable via --skip-other-mimi (saves ~6ms)
            _ = self.other_mimi.encode(chunk)

        outputs = []
        for c in range(codes.shape[-1]):
            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

            main_pcm = self.mimi.decode(tokens[:, 1:9])
            if self.other_mimi is not None:
                # Output always discarded; skippable via --skip-other-mimi (saves ~4ms)
                _ = self.other_mimi.decode(tokens[:, 1:9])

            if self._pinned_pcm is not None:
                main_pcm = main_pcm.float()
                self._pinned_pcm.copy_(main_pcm[0, 0], non_blocking=True)
                torch.cuda.current_stream().synchronize()
                out = np.array(self._pinned_pcm.numpy(), dtype=np.float32)  # copy: the buffer is reused
            else:
                out = main_pcm.float().cpu()[0, 0].numpy()

            text_token = tokens[0, 0, 0].item()
            text = None
            if text_token not in (0, 3):  # EPAD and PAD carry no words
                text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                text = text.replace("▁", " ")
            outputs.append((out, text))
        return outputs

    def new_codec(self, codec: str):
        if codec == "pcm16":
            return Pcm16Codec()
        if codec == "opus":
            return OpusCodec(self.mimi.sample_rate)
        raise ValueError(f"unsupported codec: {codec}")


def _local_or_hf(path: Optional[str], hf_repo: str, filename: str) -> str:
    """Use the local path when given. Hugging Face is only touched when no local path was given."""
    if path is not None:
        return path
    from huggingface_hub import hf_hub_download  # imported lazily: never on the local-path route

    logger.info(f"no local {filename}: fetching it from {hf_repo}")
    return hf_hub_download(hf_repo, filename)


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> str:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    from huggingface_hub import hf_hub_download  # imported lazily: never on the local-path route

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _static_path(static: Optional[str]) -> Optional[str]:
    """--static is a local directory or 'none'. Nothing is downloaded for the web UI any more."""
    if static is None or static == "none":
        return None
    if not os.path.isdir(static):
        raise SystemExit(f"--static directory does not exist: {static}")
    return static


def _ssl_context(cert_dir: str) -> ssl.SSLContext:
    """TLS from an existing cert.pem + key.pem. No mkcert download and no system trust changes."""
    cert_file = Path(cert_dir) / "cert.pem"
    key_file = Path(cert_dir) / "key.pem"
    if not (cert_file.is_file() and key_file.is_file()):
        raise SystemExit(f"--ssl needs cert.pem and key.pem in {cert_dir} (the Sim fork never downloads mkcert)")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
    return context


def _apply_fast_preset(args: argparse.Namespace) -> None:
    if not args.fast:
        return
    import importlib.util
    missing = []
    if not torch.cuda.is_available():
        missing.append("a CUDA device (required by --w8a16/--mimi-fp16)")
    if importlib.util.find_spec("triton") is None:
        missing.append("Triton (required by --w8a16's dequant GEMV and "
                       "--mimi-fp16's torch.compile path)")
    if missing:
        raise RuntimeError(
            "--fast cannot meet its performance contract; missing: "
            + "; ".join(missing) + ". No silent degradation is "
            "provided — run without --fast or install the prerequisite.")
    args.w8a16 = True
    args.dep_q_exit = args.dep_q_exit or 8
    args.skip_other_mimi = True
    args.mimi_fp16 = True


def _require_w8a16_runtime() -> None:
    """The w8a16 decode kernel is Triton on CUDA; refuse before any weight is read."""
    import importlib.util
    if not torch.cuda.is_available():
        raise SystemExit("--w8a16-checkpoint requires a CUDA device")
    if importlib.util.find_spec("triton") is None:
        raise SystemExit("--w8a16-checkpoint requires Triton (the w8a16 decode kernel)")


def _serve(args: argparse.Namespace, token: Optional[str], default_text_prompt: str) -> None:
    seed_all(42424242)
    args.device = torch_auto_device(args.device)

    ssl_context = _ssl_context(args.ssl) if args.ssl is not None else None
    static_path = _static_path(args.static)

    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if not os.path.isdir(voice_prompt_dir):
        raise SystemExit(f"voice prompt directory missing: {voice_prompt_dir}")
    voices = list_voice_files(voice_prompt_dir)
    if args.default_voice not in voices:
        raise SystemExit(f"--default-voice {args.default_voice} is not in the voice prompt directory")
    logger.info(f"voice prompts: {len(voices)} files")

    _perf_flags = (args.fast or args.fp8 or args.w8a16 or args.skip_other_mimi
                   or args.mimi_fp16 or args.pinned_io or args.dep_q_exit > 0)
    if (not _perf_flags and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 1)):
        logger.info("GB10-class device detected (sm_121): real-time "
                    "performance requires opt-in flags; try --fast "
                    "(see the GB10 PR notes)")

    logger.info("loading mimi")
    mimi_weight = _local_or_hf(args.mimi_weight, args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, args.device)
    other_mimi = None if args.skip_other_mimi else loaders.get_mimi(mimi_weight, args.device)
    if args.mimi_fp16:
        # FP16 mimi + torch.compile for faster encode/decode (~3ms saved)
        mimi = mimi.half()
        if other_mimi is not None:
            other_mimi = other_mimi.half()
        mimi.torch_compile_encoder_decoder = True
        mimi = torch.compile(mimi)
        logger.info("mimi loaded (FP16 + compiled)")
    else:
        logger.info("mimi loaded")

    tokenizer_path = _local_or_hf(args.tokenizer, args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)  # type: ignore

    logger.info("loading moshi")
    if args.w8a16_checkpoint is not None:
        # Weights are already fp8 on disk: each tensor goes to the GPU once (see w8a16_checkpoint.py).
        _require_w8a16_runtime()
        lm = load_lm_w8a16_checkpoint(args.w8a16_checkpoint, device=args.device)
        logger.info("w8a16 checkpoint loaded (weights were quantized offline)")
    else:
        moshi_weight = _local_or_hf(args.moshi_weight, args.hf_repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
        lm.eval()
        if args.fp8:
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability()
                if cc < (8, 9):
                    raise SystemExit(
                        f"--fp8 requires compute capability >= 8.9 for FP8 "
                        f"torch._scaled_mm (found sm_{cc[0]}{cc[1]})")
            from .fp8_quantize import quantize_model
            logger.info("applying FP8 quantization...")
            quantize_model(lm)
            logger.info("FP8 quantization complete")
        elif args.w8a16:
            from .w8a16_quantize import quantize_model_w8a16
            logger.info("applying w8a16 weight-only quantization...")
            quantize_model_w8a16(lm)
            logger.info("w8a16 quantization complete")
    logger.info("moshi loaded")

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        fp8=args.fp8,
        pinned_io=args.pinned_io,
        dep_q_exit=args.dep_q_exit,
    )
    logger.info("warming up the model")
    state.warmup()
    if args.fp8:
        from .fp8_quantize import free_bf16_inproj
        free_bf16_inproj(lm)

    config = SessionConfig(
        fast=args.fast,
        default_voice=args.default_voice,
        default_text_prompt=default_text_prompt,
        auth_token=token,
        max_backlog_frames=args.max_backlog_frames,
        idle_timeout_s=args.idle_timeout_s,
        max_session_s=args.max_session_seconds,
        stats_interval_s=args.stats_interval_s,
    )
    app = create_app(state, config)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    logger.info(f"model={MODEL_NAME} fast={args.fast} voices={len(voices)} "
                f"auth={'on' if token else 'off (loopback only)'}")
    logger.info(f"listening on {'https' if ssl_context else 'http'}://{args.host}:{args.port}")
    # access_log=None: request lines carry the text_prompt query, so they are never logged.
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context, access_log=None)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser(loaders.DEFAULT_REPO)
    args = parser.parse_args(argv)
    token, default_text_prompt = validate_serve_args(parser, args)
    _apply_fast_preset(args)
    if args.w8a16_checkpoint is not None:
        args.w8a16 = True  # the checkpoint is a w8a16 model; the flags that depend on it follow
    with torch.no_grad():
        _serve(args, token, default_text_prompt)


if __name__ == "__main__":
    main()
