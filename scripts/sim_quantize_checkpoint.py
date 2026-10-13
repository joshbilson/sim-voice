#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Sim fork (joshbilson/sim-voice). Offline w8a16 conversion of the PersonaPlex LM checkpoint.
"""Convert a PersonaPlex LM safetensors checkpoint to a w8a16 checkpoint for the Sim server.

The conversion runs on the CPU, one tensor at a time: peak memory is a few hundred MB above the
interpreter, never the whole model. The output holds fp8e4m3 weights, fp32 per-output-channel
scales (names as the quantized modules store them) and bf16 for everything else. The source
sha256, the conversion version and the LM configuration are recorded in the file metadata.
The server then loads it with --w8a16-checkpoint.

Example:
  OMP_NUM_THREADS=2 nice -n 10 python3 scripts/sim_quantize_checkpoint.py \\
      --src models/personaplex-7b-v1/model.safetensors \\
      --dst models/personaplex-7b-v1-w8a16/model.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(REPO_ROOT, "moshi"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert a PersonaPlex LM checkpoint to w8a16 (CPU, low memory).")
    parser.add_argument("--src", required=True, help="bf16 safetensors checkpoint of the LM (model.safetensors)")
    parser.add_argument("--dst", required=True, help="output path (written as <dst>.partial, then renamed)")
    parser.add_argument("--min-features", type=int, default=512,
                        help="same gate as the online quantizer (default 512)")
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads (default 2)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --dst")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from moshi.w8a16_checkpoint import convert_checkpoint
    from moshi.w8a16_format import CheckpointFormatError

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
        summary = convert_checkpoint(args.src, args.dst, min_features=args.min_features,
                                     threads=args.threads, overwrite=args.overwrite)
    except (OSError, ValueError, CheckpointFormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
