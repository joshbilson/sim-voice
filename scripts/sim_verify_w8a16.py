#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Sim fork (joshbilson/sim-voice). Checks a converted w8a16 checkpoint against its bf16 source.
"""Recompute the online w8a16 quantization of a few tensors and compare the stored bytes.

Samples up to --per-kind tensors each of linear, gating and in_proj (--all: every one, and every
copied bf16 tensor too). Reads only the tensors it checks.
--gpu also runs the online quantizer on CUDA and compares with the stored bytes. Exit 0 only if
every sampled tensor matches.

Example:
  python3 scripts/sim_verify_w8a16.py --source models/personaplex-7b-v1/model.safetensors \\
      --converted models/personaplex-7b-v1-w8a16/model.safetensors --gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(REPO_ROOT, "moshi"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify a w8a16 checkpoint against its source tensor by tensor.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument("--per-kind", type=int, default=3)
    parser.add_argument("--all", action="store_true",
                        help="check every quantized tensor and every copied bf16 tensor, not a sample")
    parser.add_argument("--gpu", action="store_true", help="also quantize on CUDA and compare")
    args = parser.parse_args(argv)

    import torch
    from moshi.w8a16_checkpoint import verify_converted

    device = None
    if args.gpu:
        if not torch.cuda.is_available():
            print("error: --gpu needs CUDA", file=sys.stderr)
            return 2
        device = "cuda"
    try:
        rows = verify_converted(args.source, args.converted, per_kind=None if args.all else args.per_kind,
                                device=device, include_regular=args.all)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ok = all(row["cpu_equal"] and row.get("device_equal", True) for row in rows)
    print(json.dumps({"ok": ok, "checked": len(rows), "rows": rows}, indent=2))
    return 0 if ok and rows else 1


if __name__ == "__main__":
    sys.exit(main())
