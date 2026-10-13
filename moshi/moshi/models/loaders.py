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
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path
import logging

from safetensors.torch import load_model, load_file
import torch

logger = logging.getLogger(__name__)

from ..checkpoint_plan import Copy, Expand, plan_lm_state
from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename)
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(8)
    return model


def default_lm_kwargs(delays=None) -> dict:
    """Constructor kwargs of the PersonaPlex LM (dep_q=16). A fresh dict: callers may change it."""
    lm_kwargs = dict(_lm_kwargs)
    lm_kwargs["dep_q"] = 16
    if delays is not None:
        lm_kwargs["delays"] = delays
    return lm_kwargs


def apply_checkpoint_plan(model, state_dict, *, copy_missing_weights, dev, dtype):
    """Build the state dict for model from a checkpoint's tensors, via plan_lm_state.

    Shared by get_moshi_lm and tests. Keys the model does not have are dropped; every model key
    gets a tensor on dev with dtype (copies of replicated slots and zeros included)."""
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    source_shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
    plan = plan_lm_state(model_shapes, source_shapes, copy_missing_weights=copy_missing_weights)
    out = {}
    for name, recipe in plan.items():
        if isinstance(recipe, Copy):
            tensor = state_dict[recipe.src]
        elif isinstance(recipe, Expand):
            tensor = torch.concat([state_dict[recipe.src], state_dict[recipe.src]], dim=0)
        else:
            logger.warning(f"Zero-initializing key missing from checkpoint: {name}")
            tensor = torch.zeros(model_shapes[name], dtype=dtype, device=dev)
        out[name] = tensor.to(device=dev, dtype=dtype)
    return out


def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    cpu_offload: bool = False,
) -> LMModel:
    """Return a pretrained Moshi LM model.

    Args:
        filename: Path to model weights.
        copy_missing_weights: Whether to copy missing weights from existing layers.
        device: Target device for the model.
        dtype: Data type for model weights.
        delays: Optional custom delays configuration.
        cpu_offload: If True, offload model layers to CPU when GPU memory is
                     insufficient. Uses accelerate's device_map="auto".
    """
    # Copy to avoid mutating a shared/global dict
    lm_kwargs = default_lm_kwargs(delays)

    if cpu_offload and filename is not None:
        return _get_moshi_lm_with_offload(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    # Init with meta device to avoid init dummy memory
    init_device = "meta" if filename is not None else device
    model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)
    if filename is None:
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    filename = str(filename)

    # Load state_dict
    if filename.endswith(".safetensors"):
        # safetensors does not support mps directly
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "mps":
            state_dict = load_file(filename, device="cpu")
        else:
            state_dict = load_file(filename, device=dev.type)
    else:
        # torch checkpoint
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    # Patches (expansion of depformer self-attention, replicated slots, zero-fill) and the
    # dtype cast: see checkpoint_plan.py. The dtype cast happens per tensor on dev.
    dev = torch.device(device) if isinstance(device, str) else device
    state_dict = apply_checkpoint_plan(
        model, state_dict, copy_missing_weights=copy_missing_weights, dev=dev, dtype=dtype)
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    return model.to(device=device, dtype=dtype)


def _get_moshi_lm_with_offload(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with CPU offloading using accelerate.

    This function distributes model layers across GPU and CPU based on
    available GPU memory. Layers that don't fit on GPU are kept on CPU
    and moved to GPU only during forward pass.
    """
    try:
        from accelerate import infer_auto_device_map, dispatch_model
    except ImportError:
        raise ImportError(
            "CPU offloading requires the 'accelerate' package. "
            "Install it with: pip install accelerate"
        )

    filename = str(filename)
    logger.info("Loading model with CPU offloading enabled")

    # First, create model on CPU to get the architecture
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # Load state_dict to CPU
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

    # Apply weight patches (same as non-offload path)
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")

    model.load_state_dict(state_dict, strict=False, assign=True)

    # Determine target device
    dev = torch.device(device) if isinstance(device, str) else device

    if dev.type != "cuda":
        # If not using CUDA, just move to the target device without offloading
        logger.info(f"CPU offload requested but device is {dev}, skipping offload")
        model.to(dev)
        model.eval()
        return model

    # Infer device map based on available GPU memory
    device_map = infer_auto_device_map(
        model,
        max_memory=None,  # Let accelerate auto-detect available memory
        no_split_module_classes=["StreamingTransformerLayer"],
        dtype=dtype,
    )

    # Log the device distribution
    gpu_layers = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
    cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
    logger.info(f"Device map: {gpu_layers} modules on GPU, {cpu_layers} modules on CPU")

    # Dispatch model across devices
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir="offload_weights",  # Directory for disk offload if needed
    )

    model.eval()
    return model
