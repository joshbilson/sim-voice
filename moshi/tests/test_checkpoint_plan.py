# SPDX-License-Identifier: MIT
"""Key mapping from a checkpoint to the LM state dict (torch-free; the tensor side is in
test_w8a16_checkpoint.py)."""

import unittest

import _sim_bootstrap  # noqa: F401  (must come before moshi imports)

from moshi.checkpoint_plan import (  # noqa: E402
    Copy,
    CheckpointPlanError,
    Expand,
    Zeros,
    plan_lm_state,
)


def lm_shapes(dep_q: int, n_q: int = 16, dim: int = 4):
    """Names and shapes shaped like the PersonaPlex LM, with tiny sizes. dep_q is the number of
    depformer steps: 16 for the released checkpoint, 8 for an older dep_q=8 layout."""
    shapes = {}
    for k in range(n_q):
        shapes[f"emb.{k}.weight"] = (9, dim)
    shapes["text_emb.weight"] = (11, dim)
    shapes["text_linear.weight"] = (10, dim)
    shapes["transformer.layers.0.self_attn.in_proj_weight"] = (3 * dim, dim)
    shapes["transformer.layers.0.self_attn.out_proj.weight"] = (dim, dim)
    shapes["transformer.layers.0.gating.linear_in.weight"] = (2 * dim, dim)
    shapes["transformer.layers.0.norm1.alpha"] = (1, 1, dim)
    for k in range(dep_q):
        shapes[f"depformer_in.{k}.weight"] = (dim, dim)
        shapes[f"linears.{k}.weight"] = (10, dim)
        shapes[f"depformer.layers.0.gating.{k}.linear_in.weight"] = (6, dim)
    for k in range(dep_q - 1):
        shapes[f"depformer_emb.{k}.weight"] = (9, dim)
    shapes["depformer.layers.0.self_attn.in_proj_weight"] = (3 * dim * dep_q, dim)
    shapes["depformer.layers.0.self_attn.out_proj.weight"] = (dim * dep_q, dim)
    return shapes


class PlanForReleasedLayout(unittest.TestCase):
    def test_identical_layout_is_a_plain_copy(self):
        model = lm_shapes(dep_q=16)
        plan = plan_lm_state(model, model)
        self.assertEqual(list(plan), list(model))  # model order
        for name, recipe in plan.items():
            self.assertEqual(recipe, Copy(name))

    def test_unknown_source_keys_are_dropped(self):
        model = lm_shapes(dep_q=16)
        source = dict(model)
        source["some.legacy.tensor"] = (2, 2)
        plan = plan_lm_state(model, source)
        self.assertNotIn("some.legacy.tensor", plan)
        self.assertEqual(set(plan), set(model))

    def test_missing_model_key_without_a_replica_is_zero_filled(self):
        model = lm_shapes(dep_q=16)
        source = dict(model)
        del source["text_emb.weight"]
        plan = plan_lm_state(model, source)
        self.assertEqual(plan["text_emb.weight"], Zeros())


class PlanForDepq8Source(unittest.TestCase):
    """A dep_q=8 checkpoint loaded into a dep_q=16 model: the three historical patches."""

    def setUp(self):
        self.model = lm_shapes(dep_q=16)
        self.source = lm_shapes(dep_q=8)
        self.plan = plan_lm_state(self.model, self.source)

    def test_depformer_self_attention_is_expanded(self):
        self.assertEqual(self.plan["depformer.layers.0.self_attn.in_proj_weight"],
                         Expand("depformer.layers.0.self_attn.in_proj_weight"))
        self.assertEqual(self.plan["depformer.layers.0.self_attn.out_proj.weight"],
                         Expand("depformer.layers.0.self_attn.out_proj.weight"))

    def test_replicated_slots_copy_from_the_lower_slot(self):
        self.assertEqual(self.plan["linears.8.weight"], Copy("linears.0.weight"))
        self.assertEqual(self.plan["linears.15.weight"], Copy("linears.7.weight"))
        self.assertEqual(self.plan["depformer_in.9.weight"], Copy("depformer_in.1.weight"))
        self.assertEqual(self.plan["depformer.layers.0.gating.12.linear_in.weight"],
                         Copy("depformer.layers.0.gating.4.linear_in.weight"))
        self.assertEqual(self.plan["depformer_emb.10.weight"], Copy("depformer_emb.2.weight"))

    def test_slot_without_a_source_is_zero(self):
        # depformer_emb has dep_q-1 = 15 entries; index 7 is never a replica target (only 8..15 are).
        self.assertEqual(self.plan["depformer_emb.7.weight"], Zeros())
        self.assertEqual(self.plan["depformer_emb.14.weight"], Copy("depformer_emb.6.weight"))
        self.assertNotIn("depformer_emb.15.weight", self.plan)  # the model has dep_q-1 = 15 of them

    def test_direct_keys_are_still_copies(self):
        self.assertEqual(self.plan["emb.3.weight"], Copy("emb.3.weight"))
        self.assertEqual(self.plan["transformer.layers.0.self_attn.out_proj.weight"],
                         Copy("transformer.layers.0.self_attn.out_proj.weight"))

    def test_every_model_key_is_planned(self):
        self.assertEqual(set(self.plan), set(self.model))


class PlanRefusals(unittest.TestCase):
    def test_expansion_that_does_not_fit_is_refused(self):
        model = lm_shapes(dep_q=16)
        source = lm_shapes(dep_q=8)
        source["depformer.layers.0.self_attn.out_proj.weight"] = (5, 4)
        with self.assertRaises(CheckpointPlanError):
            plan_lm_state(model, source)

    def test_expansion_is_refused_without_copy_missing_weights(self):
        with self.assertRaises(CheckpointPlanError):
            plan_lm_state(lm_shapes(dep_q=16), lm_shapes(dep_q=8), copy_missing_weights=False)

    def test_copy_with_a_different_shape_is_refused(self):
        model = lm_shapes(dep_q=16)
        source = dict(model)
        source["text_linear.weight"] = (10, 5)
        with self.assertRaises(CheckpointPlanError):
            plan_lm_state(model, source)

    def test_without_copy_missing_weights_absent_slots_only_zero_fill(self):
        model = lm_shapes(dep_q=16)
        source = {k: v for k, v in model.items() if not any(f"linears.{i}." in k for i in range(8, 16))}
        plan = plan_lm_state(model, source, copy_missing_weights=False)
        self.assertEqual(plan["linears.8.weight"], Zeros())
        self.assertEqual(plan["linears.3.weight"], Copy("linears.3.weight"))


if __name__ == "__main__":
    unittest.main()
