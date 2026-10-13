# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# Command-line flags and start-up checks for the Sim voice server. Torch-free, so the refusal
# rules (non-loopback host needs a token file, no gradio tunnel, token file permissions, sane
# limits) are tested without a GPU. The flags are documented in SIM.md.

from __future__ import annotations

import argparse
from typing import Optional

from .sim_protocol import (
    DEFAULT_VOICE,
    VOICE_RE,
    TokenFileError,
    is_loopback_host,
    load_token_file,
    read_default_text_prompt,
)
from .w8a16_format import CheckpointFormatError, check_w8a16_metadata, read_header

DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"  # same value as moshi.models.loaders.DEFAULT_REPO


def build_arg_parser(default_hf_repo: str = DEFAULT_HF_REPO) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sim voice server: PersonaPlex, one call at a time.")
    parser.add_argument("--host", default="localhost", type=str,
                        help="Bind address. A non-loopback address requires --auth-token-file.")
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", default="none", type=str,
                        help="Local directory with a web UI to serve at /, or 'none' (default).")
    parser.add_argument("--gradio-tunnel", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gradio-tunnel-token", help=argparse.SUPPRESS)
    parser.add_argument("--auth-token-file", type=str,
                        help="File holding the bearer token (chmod 600). Every /api/* request needs "
                             "'Authorization: Bearer <token>'; a wrong or missing token gets HTTP 401.")
    parser.add_argument("--default-voice", type=str, default=DEFAULT_VOICE,
                        help="Voice used when a call does not name one (must be in the voice dir).")
    parser.add_argument("--default-text-prompt-file", type=str,
                        help="File with the default text prompt (<= 1200 characters).")
    parser.add_argument("--max-backlog-frames", type=int, default=6,
                        help="Input frames (80 ms each) allowed to wait; older ones are dropped.")
    parser.add_argument("--idle-timeout-s", type=float, default=15.0,
                        help="End a call (timeout, 4408) after this long without client audio.")
    parser.add_argument("--max-session-seconds", type=float, default=1800.0,
                        help="End a call (timeout, 4408) after this many seconds.")
    parser.add_argument("--stats-interval-s", type=float, default=2.0,
                        help="Seconds between stats messages (0x06).")
    parser.add_argument("--ssl", type=str,
                        help="Directory with an existing cert.pem and key.pem. Nothing is downloaded.")

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=default_hf_repo,
                        help="HF repo used only for files that have no local path. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Quantize LM weights to FP8 for ~1.4x faster inference (requires SM >= 89)"
    )
    parser.add_argument(
        "--skip-other-mimi", action="store_true",
        help="Skip the second mimi stream (its outputs are discarded in this serve path)")
    parser.add_argument(
        "--mimi-fp16", action="store_true",
        help="Run mimi in fp16 with torch.compile (requires a working torch.compile backend)")
    parser.add_argument(
        "--pinned-io", action="store_true",
        help="Use a pinned CPU buffer for DtoH audio transfer")
    parser.add_argument(
        "--w8a16", action="store_true",
        help="Weight-only 8-bit LM quantization, bf16 compute (Triton GEMV); exclusive with --fp8")
    parser.add_argument(
        "--w8a16-checkpoint", type=str, default=None,
        help="LM checkpoint already quantized to w8a16 by scripts/sim_quantize_checkpoint.py. Loaded "
             "tensor by tensor, so start-up peaks near the final footprint. Implies --w8a16 and "
             "replaces --moshi-weight.")
    parser.add_argument(
        "--fast", action="store_true",
        help="Preset: --w8a16 --dep-q-exit 8 --skip-other-mimi --mimi-fp16")
    parser.add_argument(
        "--dep-q-exit", type=int, default=0,
        help="Stop the depformer after N steps (>=8). Safe in the serve flow "
             "because user-side codebooks are always provided.")
    return parser


def validate_serve_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Optional[str], str]:
    """Start-up checks that need no model. Exits through parser.error on a refusal.

    Returns (bearer token or None when loopback-only, default text prompt).
    """
    if args.gradio_tunnel or args.gradio_tunnel_token is not None:
        parser.error("--gradio-tunnel was removed in the Sim fork: no third-party tunnels")
    if args.fp8 and args.w8a16:
        parser.error("--fp8 and --w8a16 are mutually exclusive")
    if args.w8a16_checkpoint is not None:
        if args.fp8:
            parser.error("--w8a16-checkpoint and --fp8 are mutually exclusive")
        if args.cpu_offload:
            parser.error("--w8a16-checkpoint cannot be combined with --cpu-offload")
        if args.moshi_weight is not None:
            parser.error("pass --w8a16-checkpoint or --moshi-weight, not both")
        try:
            check_w8a16_metadata(read_header(args.w8a16_checkpoint).metadata)
        except CheckpointFormatError as exc:
            parser.error(f"--w8a16-checkpoint: {exc}")
    if not VOICE_RE.fullmatch(args.default_voice):
        parser.error("--default-voice must look like NATF2.pt")
    if args.max_backlog_frames < 1:
        parser.error("--max-backlog-frames must be at least 1")
    if args.idle_timeout_s <= 0 or args.max_session_seconds <= 0 or args.stats_interval_s <= 0:
        parser.error("--idle-timeout-s, --max-session-seconds and --stats-interval-s must be positive")

    token = None
    if args.auth_token_file is not None:
        try:
            token = load_token_file(args.auth_token_file)
        except TokenFileError as exc:
            parser.error(str(exc))
    if token is None and not is_loopback_host(args.host):
        parser.error("--host is not a loopback address, so --auth-token-file is required")

    try:
        default_text = read_default_text_prompt(args.default_text_prompt_file)
    except (OSError, ValueError) as exc:
        parser.error(f"--default-text-prompt-file is unusable ({type(exc).__name__}: {exc})")
    return token, default_text
