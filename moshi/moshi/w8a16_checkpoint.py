# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# Offline w8a16 conversion and low-peak loading of the LM.
#
# Why: the online path loads the whole bf16 checkpoint onto the GPU and then quantizes it, so the
# two copies coexist (about 24 GB extra on a 16.7 GB model). convert_checkpoint quantizes on the
# CPU one tensor at a time into a file that already holds the fp8 weights and fp32 scales.
# load_lm_w8a16_checkpoint then places each stored tensor on the GPU once, so the start-up peak
# is roughly the final footprint.
#
# Numerics: the stored fp8 and scale bytes equal what quantize_model_w8a16 computes on the GPU
# from the same bf16 weights. The row-chunked quantization below performs the same per-row
# operations, so the results are bit-identical; tests/test_w8a16_checkpoint.py and
# scripts/sim_verify_w8a16.py check it.

from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import shutil
import time
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

from .models import loaders
from .checkpoint_plan import Copy, Expand, plan_lm_state
from .models.lm import LMModel
from .w8a16_format import (
    CONVERSION_VERSION,
    ITEMSIZE,
    METADATA_PREFIX,
    CheckpointFormatError,
    SafetensorsWriter,
    check_w8a16_metadata,
    read_header,
    sha256_file,
)
from .w8a16_quantize import attach_w8a16_structure, w8a16_targets

logger = logging.getLogger(__name__)

ROW_CHUNK = 1024  # rows per quantization step: bounds the fp32 temporaries to a few tens of MB
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
ST_NAME = {BF16: "BF16", torch.float32: "F32", FP8: "F8_E4M3"}
FREE_DISK_MARGIN = 1024 ** 3


def row_scales(w: torch.Tensor, chunk: int = ROW_CHUNK) -> torch.Tensor:
    """Per-output-channel scales, the same numbers as w8a16_quantize._quantize_weight."""
    amax = torch.empty(w.shape[0], dtype=torch.float32)
    for start in range(0, w.shape[0], chunk):
        amax[start:start + chunk] = w[start:start + chunk].abs().amax(dim=1).float()
    return (amax / 448.0).clamp(min=1e-12)


def iter_fp8_rows(w: torch.Tensor, scale: torch.Tensor, chunk: int = ROW_CHUNK):
    """fp8 weight in row chunks; concatenated, equal to _quantize_weight(w)[0]."""
    for start in range(0, w.shape[0], chunk):
        yield (w[start:start + chunk].float() / scale[start:start + chunk, None]).to(FP8)


def _raw(tensor: torch.Tensor) -> memoryview:
    """Little-endian bytes of a CPU tensor, without a copy when it is contiguous."""
    flat = tensor.contiguous().view(torch.uint8).numpy().reshape(-1)
    return memoryview(np.ascontiguousarray(flat))


def _st_dtype(dtype: torch.dtype) -> str:
    if dtype not in ST_NAME:
        raise CheckpointFormatError(f"no safetensors name for {dtype}")
    return ST_NAME[dtype]


def _source_tensor(handle, recipe, shape) -> torch.Tensor:
    if isinstance(recipe, Copy):
        tensor = handle.get_tensor(recipe.src)
    elif isinstance(recipe, Expand):
        base = handle.get_tensor(recipe.src)
        tensor = torch.concat([base, base], dim=0)
        del base
    else:
        tensor = torch.zeros(shape, dtype=BF16)
    return tensor.to(BF16)


def convert_checkpoint(
    src: str,
    dst: str,
    *,
    lm_kwargs: Optional[dict] = None,
    min_features: int = 512,
    copy_missing_weights: bool = True,
    threads: int = 2,
    overwrite: bool = False,
) -> dict:
    """Quantize a PersonaPlex LM safetensors checkpoint into a w8a16 checkpoint at dst.

    Peak memory is a few hundred MB above the interpreter: one source tensor and one row chunk
    at a time. The output is written to dst + '.partial' and renamed when complete.
    """
    started = time.monotonic()
    if threads:
        torch.set_num_threads(threads)
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"{dst} exists (pass overwrite=True to replace it)")

    header = read_header(src)
    src_shapes = {name: info.shape for name, info in header.tensors.items()}
    kwargs = dict(lm_kwargs) if lm_kwargs is not None else loaders.default_lm_kwargs()

    model = LMModel(device="meta", dtype=BF16, **kwargs)
    pre = {name: (tuple(v.shape)) for name, v in model.state_dict().items()}
    plan = plan_lm_state(pre, src_shapes, copy_missing_weights=copy_missing_weights)
    targets = w8a16_targets(model, min_features)
    attach_w8a16_structure(model, min_features)
    post = {name: (tuple(v.shape), v.dtype) for name, v in model.state_dict().items()}

    # Groups: a quantized module writes its outputs together, from one source tensor.
    groups = []
    grouped = set()
    for name, kind in targets:
        if kind == "linear":
            source, outputs = f"{name}.weight", [f"{name}.weight", f"{name}.w8a16_scale"]
        else:
            source = f"{name}.in_proj_weight"
            outputs = [f"{name}._in_proj_q_weight", f"{name}._in_proj_q_scale", f"{name}.in_proj_weight"]
        groups.append((kind, source, outputs))
        grouped.update(outputs)
    regular = sorted(name for name in post if name not in grouped)
    for name in regular:
        if name not in plan or post[name][1] != BF16:
            raise CheckpointFormatError(f"{name} is neither quantized nor a bf16 model tensor")

    entries = []  # physical order: (name, st dtype, shape, nbytes)
    items = []
    for name in regular:
        shape, dtype = post[name]
        nbytes = int(np.prod(shape, dtype=np.int64)) * 2
        items.append((name, "regular", None, [(name, _st_dtype(dtype), shape, nbytes)]))
    for kind, source, outputs in groups:
        group_entries = []
        for out in outputs:
            shape, dtype = post[out]
            nbytes = int(np.prod(shape, dtype=np.int64)) * ITEMSIZE[_st_dtype(dtype)]
            group_entries.append((out, _st_dtype(dtype), shape, nbytes))
        items.append((outputs[0], kind, source, group_entries))
    items.sort(key=lambda item: item[0])
    for _, _, _, group_entries in items:
        entries.extend(group_entries)

    total = sum(entry[3] for entry in entries)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(dst))).free
    if free < total + FREE_DISK_MARGIN:
        raise OSError(f"not enough free disk: need {total} bytes plus margin, have {free}")

    src_sha = sha256_file(src)
    metadata = {
        "format": "pt",
        METADATA_PREFIX + "version": CONVERSION_VERSION,
        METADATA_PREFIX + "source_sha256": src_sha,
        METADATA_PREFIX + "source_name": os.path.basename(src),
        METADATA_PREFIX + "min_features": str(int(min_features)),
        METADATA_PREFIX + "copy_missing_weights": "1" if copy_missing_weights else "0",
        METADATA_PREFIX + "lm_kwargs": json.dumps(kwargs, sort_keys=True),
    }
    n_lin = sum(1 for kind, _, _ in groups if kind == "linear")
    n_inproj = len(groups) - n_lin
    logger.info("converting %d tensors (%d linear, %d in_proj) to %s", len(entries), n_lin, n_inproj,
                os.path.basename(dst))

    writer = SafetensorsWriter(dst, entries, metadata)
    try:
        with safe_open(src, framework="pt", device="cpu") as handle:
            for name, kind, source, group_entries in items:
                if kind == "regular":
                    tensor = _source_tensor(handle, plan[name], post[name][0])
                    writer.write(name, _raw(tensor))
                    del tensor
                    continue
                weight = _source_tensor(handle, plan[source], pre[source])
                scale = row_scales(weight)
                for out, st_dtype, _, _ in group_entries:
                    if st_dtype == "F8_E4M3":
                        for chunk in iter_fp8_rows(weight, scale):
                            writer.write(out, _raw(chunk))
                    elif st_dtype == "F32":
                        writer.write(out, _raw(scale))
                    else:  # the empty placeholder that replaces the bf16 in_proj_weight
                        writer.write(out, b"")
                del weight, scale
        out_sha = writer.finish()
    except BaseException:
        writer.abort()
        raise

    _check_written(dst, entries, metadata)
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "source": os.path.basename(src),
        "output": os.path.basename(dst),
        "conversion_version": CONVERSION_VERSION,
        "source_sha256": src_sha,
        "output_sha256": out_sha,
        "tensors": len(entries),
        "linear_targets": n_lin,
        "in_proj_targets": n_inproj,
        "data_bytes": total,
        "seconds": round(time.monotonic() - started, 1),
        "max_rss_mib": round(rss_mib, 1),
    }


def _check_written(path: str, entries: list, metadata: dict) -> None:
    header = read_header(path)
    for name, dtype, shape, nbytes in entries:
        info = header.tensors.get(name)
        if info is None or info.dtype != dtype or info.shape != tuple(shape) or info.nbytes != nbytes:
            raise CheckpointFormatError(f"written file disagrees with the plan at {name}")
    if set(header.tensors) != {e[0] for e in entries}:
        raise CheckpointFormatError("written file has unexpected tensors")
    if header.metadata.get(METADATA_PREFIX + "version") != metadata[METADATA_PREFIX + "version"]:
        raise CheckpointFormatError("written file has the wrong metadata")


def load_lm_w8a16_checkpoint(path: str, *, device="cuda", lm_kwargs: Optional[dict] = None) -> LMModel:
    """Build the quantized LM from a converted checkpoint, one stored tensor at a time.

    The model is created on the meta device with the w8a16 layout, so no bf16 copy of any
    weight is ever materialized. Every stored tensor is copied to device once and assigned.
    """
    header = read_header(path)
    check_w8a16_metadata(header.metadata)
    kwargs = dict(lm_kwargs) if lm_kwargs is not None else loaders.default_lm_kwargs()
    if header.metadata.get(METADATA_PREFIX + "lm_kwargs") != json.dumps(kwargs, sort_keys=True):
        raise CheckpointFormatError("the w8a16 checkpoint was made for another LM configuration")
    min_features = int(header.metadata[METADATA_PREFIX + "min_features"])

    model = LMModel(device="meta", dtype=BF16, **kwargs)
    attach_w8a16_structure(model, min_features)
    expected = model.state_dict()

    problems = []
    for name, value in expected.items():
        info = header.tensors.get(name)
        if info is None:
            problems.append(f"missing {name}")
        elif info.shape != tuple(value.shape) or info.dtype != _st_dtype(value.dtype):
            problems.append(f"{name}: file {info.dtype}{list(info.shape)}")
    extra = sorted(set(header.tensors) - set(expected))
    problems += [f"unexpected {name}" for name in extra]
    if problems:
        raise CheckpointFormatError("w8a16 checkpoint does not match the model: " + "; ".join(problems[:5]))

    dev = torch.device(device)
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in expected:
            tensors[name] = handle.get_tensor(name).to(dev)
    model.load_state_dict(tensors, strict=True, assign=True)
    del tensors
    if any(t.device.type != dev.type for t in list(model.parameters()) + list(model.buffers())):
        raise CheckpointFormatError("a tensor stayed on the meta device")
    model.eval()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return model


def _evenly(items: list, count: int) -> list:
    if len(items) <= count:
        return list(items)
    step = (len(items) - 1) / (count - 1) if count > 1 else 0
    return [items[round(i * step)] for i in range(count)]


def verify_converted(source: str, converted: str, *, per_kind: int = 3, device=None) -> list[dict]:
    """Compare stored w8a16 tensors with the online quantizer applied to the source.

    Samples up to per_kind tensors each of 'linear' (non-gating nn.Linear), 'gating' and 'in_proj',
    recomputes the fp8 weight and fp32 scale from the bf16 source with _quantize_weight, and compares
    the bytes. With device set, the same quantization also runs on that device and must match too.
    Reads only the sampled tensors, never the whole model.
    """
    from .w8a16_quantize import _quantize_weight

    header = read_header(converted)
    check_w8a16_metadata(header.metadata)
    source_header = read_header(source)
    fp8_names = [n for n, info in header.tensors.items() if info.dtype == "F8_E4M3"]

    candidates: dict[str, list] = {"linear": [], "gating": [], "in_proj": []}
    for name in sorted(fp8_names):
        if name.endswith("._in_proj_q_weight"):
            module = name[: -len("._in_proj_q_weight")]
            candidates["in_proj"].append((name, f"{module}.in_proj_weight", f"{module}._in_proj_q_scale"))
        elif name.endswith(".weight") and f"{name[:-len('.weight')]}.w8a16_scale" in header.tensors:
            module = name[: -len(".weight")]
            kind = "gating" if ".gating." in name else "linear"
            candidates[kind].append((name, name, f"{module}.w8a16_scale"))

    rows = []
    with safe_open(source, framework="pt", device="cpu") as src_f, \
            safe_open(converted, framework="pt", device="cpu") as conv_f:
        for kind, items in candidates.items():
            usable = [item for item in items if item[1] in source_header.tensors]
            for stored_name, source_key, scale_name in _evenly(usable, per_kind):
                weight = src_f.get_tensor(source_key).to(BF16)
                ref_fp8, ref_scale = _quantize_weight(weight)
                stored_fp8 = conv_f.get_tensor(stored_name)
                stored_scale = conv_f.get_tensor(scale_name)
                fp8_equal = torch.equal(ref_fp8.view(torch.uint8), stored_fp8.view(torch.uint8))
                scale_equal = torch.equal(ref_scale.view(torch.int32), stored_scale.view(torch.int32))
                row = {
                    "kind": kind,
                    "tensor": stored_name,
                    "shape": list(stored_fp8.shape),
                    "fp8_bytes": int(stored_fp8.numel()),
                    "fp8_sha256_16": hashlib.sha256(stored_fp8.view(torch.uint8).numpy().tobytes()).hexdigest()[:16],
                    "cpu_equal": bool(fp8_equal and scale_equal),
                }
                if device is not None:
                    gpu_fp8, gpu_scale = _quantize_weight(weight.to(device))
                    row["device_equal"] = bool(
                        torch.equal(gpu_fp8.cpu().view(torch.uint8), stored_fp8.view(torch.uint8))
                        and torch.equal(gpu_scale.cpu().view(torch.int32), stored_scale.view(torch.int32)))
                rows.append(row)
    return rows
