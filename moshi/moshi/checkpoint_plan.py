# SPDX-License-Identifier: MIT
#
# Sim fork additions (joshbilson/sim-voice). Licensed like the rest of this repository
# (see LICENSE-MIT and moshi/LICENSE.moshi).
#
# Which source tensor fills each key of the LM state dict. This is the three-step patch that
# loaders.get_moshi_lm has always applied to a checkpoint (depformer self-attention expansion,
# copying the 0..7 slots of per-codebook layers into 8..15, zero-filling what is still missing),
# written as pure data over names and shapes. The online loader and the offline w8a16 converter
# (w8a16_checkpoint.py) both execute this one plan, so they cannot disagree about a key.
#
# Torch-free: nothing here touches tensors, so the plan is tested without a GPU.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Union

REPLICATED_GROUPS = ("gating", "linears", "depformer_in", "depformer_emb")
REPLICA_SLOTS = tuple(zip(range(8), range(8, 16)))  # (source index, target index)


@dataclass(frozen=True)
class Copy:
    """The target is the source tensor, unchanged (a shared alias until the dtype cast)."""

    src: str


@dataclass(frozen=True)
class Expand:
    """The target is the source tensor stacked on itself along dim 0 (depformer self-attention
    is per-step, so a dep_q=8 checkpoint is doubled to 16 steps)."""

    src: str


@dataclass(frozen=True)
class Zeros:
    """Absent from the checkpoint and not derivable from it: zero-initialised."""


Recipe = Union[Copy, Expand, Zeros]


class CheckpointPlanError(ValueError):
    """The source checkpoint cannot be mapped onto the model without guessing."""


def _replica_source(name: str, source_shapes: Mapping[str, tuple]) -> Optional[str]:
    # Same search as the original loader: for each (old, new) slot, the first replicated group
    # whose "<group>.<new>." appears in the name decides; a missing source moves on to the next slot.
    for old, new in REPLICA_SLOTS:
        for group in REPLICATED_GROUPS:
            needle = f"{group}.{new}."
            if needle in name:
                src = name.replace(needle, f"{group}.{old}.")
                if src in source_shapes:
                    return src
                break
    return None


def plan_lm_state(
    model_shapes: Mapping[str, tuple],
    source_shapes: Mapping[str, tuple],
    *,
    copy_missing_weights: bool = True,
) -> dict[str, Recipe]:
    """Map every key of the model's state dict to the recipe that produces its tensor.

    model_shapes: name -> shape of the (pre-quantization) model state dict.
    source_shapes: name -> shape of the checkpoint. Checkpoint keys the model does not have are
    dropped, as load_state_dict(strict=False) always did.

    Returns a dict with exactly the model's keys, in model order.
    """
    model_shapes = {k: tuple(v) for k, v in model_shapes.items()}
    source_shapes = {k: tuple(v) for k, v in source_shapes.items()}

    # Patch 1: depformer self-attention (in_proj and out_proj) is expanded when the checkpoint has fewer steps.
    expanded: set[str] = set()
    for name, shape in source_shapes.items():
        if "depformer" not in name or "self_attn" not in name or name not in model_shapes:
            continue
        want = model_shapes[name]
        if shape == want:
            continue
        if not copy_missing_weights:
            raise CheckpointPlanError(f"{name} has shape {shape}, model wants {want}, and copy_missing_weights is off")
        if len(want) != len(shape) or want[1:] != shape[1:] or want[0] != 2 * shape[0]:
            raise CheckpointPlanError(f"{name} has shape {shape}; expanding it cannot give {want}")
        expanded.add(name)

    plan: dict[str, Recipe] = {}
    for name in model_shapes:
        if name in source_shapes:
            plan[name] = Expand(name) if name in expanded else Copy(name)

    # Patch 2: keys the checkpoint lacks are copied from the matching lower slot of a replicated group.
    if copy_missing_weights:
        for name in model_shapes:
            if name in plan:
                continue
            src = _replica_source(name, source_shapes)
            if src is not None:
                plan[name] = Copy(src)

    # Patch 3: anything still missing is zero-initialised, so no key stays on the meta device.
    for name in model_shapes:
        if name not in plan:
            plan[name] = Zeros()

    for name, recipe in plan.items():
        if isinstance(recipe, Copy) and source_shapes[recipe.src] != model_shapes[name]:
            raise CheckpointPlanError(
                f"{recipe.src} has shape {source_shapes[recipe.src]}, model wants {model_shapes[name]} for {name}")
    return plan
