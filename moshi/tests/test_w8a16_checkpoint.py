# SPDX-License-Identifier: MIT
"""Offline w8a16 conversion and low-peak loading, checked on a tiny LM on the CPU.

What is proven here: (1) the stored fp8 weights and fp32 scales are bit-identical to the online
quantizer, (2) a model loaded from the converted file has the same state dict and the same
forward outputs as the model quantized online from the same source, both for the released
dep_q=16 layout and for an older dep_q=8 layout that exercises the loader's patches, (3) the
refusals. Needs torch and safetensors; skipped without them. No GPU and no weights.

Set W8A16_SOURCE to a real bf16 LM checkpoint and W8A16_CONVERTED to its converted file to also
run the sampled real-tensor check (test_real_sample_matches_source); it reads only a few tensors.
"""

import os
import tempfile
import unittest
from unittest import mock

# The lazy torch.compile wrappers read this at import time; the CPU tests must not compile.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import _sim_bootstrap  # noqa: F401,E402  (must come before moshi imports)

try:
    import torch
    from safetensors.torch import save_file
    HAVE_TORCH = True
except ImportError:  # the torch-free suite still runs
    HAVE_TORCH = False

if HAVE_TORCH:
    from moshi import w8a16_checkpoint as wc
    from moshi import w8a16_quantize as wq
    from moshi.models import loaders
    from moshi.models.lm import LMModel
    from moshi.w8a16_format import METADATA_PREFIX, CheckpointFormatError, read_header

BF16 = torch.bfloat16 if HAVE_TORCH else None
MIN_FEATURES = 16  # small enough that the tiny model has quantized layers of every kind


def tiny_kwargs(dep_q: int = 16) -> dict:
    """The constructor kwargs of loaders._lm_kwargs at tiny sizes (same keys, same structure)."""
    return dict(
        delays=[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        n_q=16, dep_q=dep_q, card=16, text_card=32, existing_text_padding_id=3,
        dim=32, num_heads=4, num_layers=2, hidden_scale=4.125, causal=True, layer_scale=None,
        context=16, max_period=10000, gating="silu", norm="rms_norm_f32", positional_embedding="rope",
        depformer_dim=16, depformer_dim_feedforward=66, depformer_num_heads=4, depformer_num_layers=1,
        depformer_causal=True, depformer_layer_scale=None, depformer_multi_linear=True,
        depformer_context=8, depformer_max_period=10000, depformer_gating="silu",
        depformer_pos_emb="none", depformer_weights_per_step=True,
    )


def random_lm_state(kwargs: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    model = LMModel(device="cpu", dtype=BF16, **kwargs)
    model.eval()
    return {k: v.detach().contiguous() for k, v in model.state_dict().items()}


def online_quantized(kwargs: dict, source_state: dict, min_features: int = MIN_FEATURES) -> LMModel:
    """The reference: what get_moshi_lm + quantize_model_w8a16 produce, minus the CUDA checks."""
    model = LMModel(device="meta", dtype=BF16, **kwargs)
    state = loaders.apply_checkpoint_plan(
        model, source_state, copy_missing_weights=True, dev=torch.device("cpu"), dtype=BF16)
    model.load_state_dict(state, strict=False, assign=True)
    model.eval()
    wq.install_forward_patches()
    for name, kind in wq.w8a16_targets(model, min_features):
        module = model.get_submodule(name)
        if kind == "linear":
            wq.quantize_linear_w8a16(module)
        else:
            wq.quantize_in_proj_w8a16(module)
    return model


def assert_same_state(test: unittest.TestCase, a: LMModel, b: LMModel) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    test.assertEqual(set(sa), set(sb))
    for name in sa:
        x, y = sa[name], sb[name]
        test.assertEqual(x.dtype, y.dtype, name)
        test.assertEqual(tuple(x.shape), tuple(y.shape), name)
        test.assertEqual(x.device.type, "cpu", name)
        test.assertTrue(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)),
                        f"{name} differs")


def step_outputs(model: LMModel, seed: int) -> list:
    """One main-transformer step and one full depformer pass, outputs as float tensors."""
    gen = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, model.card, (1, model.num_codebooks, 1), generator=gen)
    seq[:, 0] = torch.randint(0, model.text_card, (1, 1), generator=gen)
    outs = []
    with torch.no_grad():
        transformer_out, text_logits = model.forward_codes(seq)
        outs.append(text_logits.float())
        with model.depformer.streaming(1):
            for cb in range(model.dep_q):
                logits = model.forward_depformer(cb, seq[:, :1, :] if cb == 0 else seq[:, cb:cb + 1, :],
                                                 transformer_out)
                outs.append(logits.float())
    return outs


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class QuantizationMatchesOnline(unittest.TestCase):
    def test_row_chunked_quantization_is_bit_identical(self):
        torch.manual_seed(1)
        w = (torch.randn(1000, 300) * 0.07).to(BF16)
        w[17] = 0  # a zero row exercises the clamp on the scale
        w[400] = w[400] * 9000  # a large row
        ref_fp8, ref_scale = wq._quantize_weight(w)
        scale = wc.row_scales(w, chunk=128)
        fp8 = torch.cat(list(wc.iter_fp8_rows(w, scale, chunk=128)))
        self.assertTrue(torch.equal(scale.view(torch.int32), ref_scale.view(torch.int32)))
        self.assertTrue(torch.equal(fp8.view(torch.uint8), ref_fp8.view(torch.uint8)))

    def test_targets_cover_each_kind_and_skip_depformer_attention(self):
        model = LMModel(device="meta", dtype=BF16, **tiny_kwargs())
        targets = dict(wq.w8a16_targets(model, MIN_FEATURES))
        self.assertEqual(targets["text_linear"], "linear")
        self.assertEqual(targets["depformer_in.3"], "linear")
        self.assertEqual(targets["linears.15"], "linear")
        self.assertEqual(targets["transformer.layers.0.gating.linear_in"], "linear")
        self.assertEqual(targets["depformer.layers.0.gating.5.linear_out"], "linear")
        self.assertEqual(targets["transformer.layers.1.self_attn"], "in_proj")
        self.assertNotIn("depformer.layers.0.self_attn.out_proj", targets)
        self.assertNotIn("depformer.layers.0.self_attn", targets)
        self.assertNotIn("emb.0", targets)  # embeddings are not Linear: they stay bf16


@unittest.skipUnless(HAVE_TORCH, "needs torch")
class ConvertedCheckpoint(unittest.TestCase):
    """Converted file vs online quantization of the same source, for both source layouts."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.root = cls.dir.name
        cls.kwargs = tiny_kwargs(16)
        cls.src16 = os.path.join(cls.root, "src16.safetensors")
        cls.src8 = os.path.join(cls.root, "src8.safetensors")
        cls.state16 = random_lm_state(cls.kwargs, seed=3)
        save_file(cls.state16, cls.src16)
        cls.state8 = random_lm_state(tiny_kwargs(8), seed=4)
        save_file(cls.state8, cls.src8)
        cls.summary16 = wc.convert_checkpoint(cls.src16, os.path.join(cls.root, "w16.safetensors"),
                                              lm_kwargs=cls.kwargs, min_features=MIN_FEATURES, threads=1)
        cls.summary8 = wc.convert_checkpoint(cls.src8, os.path.join(cls.root, "w8.safetensors"),
                                             lm_kwargs=cls.kwargs, min_features=MIN_FEATURES, threads=1)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_released_layout_loads_identically_to_online(self):
        loaded = wc.load_lm_w8a16_checkpoint(os.path.join(self.root, "w16.safetensors"),
                                             device="cpu", lm_kwargs=self.kwargs)
        online = online_quantized(self.kwargs, self.state16)
        assert_same_state(self, loaded, online)
        for a, b in zip(step_outputs(loaded, seed=7), step_outputs(online, seed=7)):
            self.assertTrue(torch.equal(a, b))

    def test_dep_q8_layout_exercises_the_patches_and_still_matches(self):
        loaded = wc.load_lm_w8a16_checkpoint(os.path.join(self.root, "w8.safetensors"),
                                             device="cpu", lm_kwargs=self.kwargs)
        online = online_quantized(self.kwargs, self.state8)
        assert_same_state(self, loaded, online)
        # The replicated slots really were copied from the lower slot, and the expansion doubled 8 -> 16.
        sd = loaded.state_dict()
        self.assertTrue(torch.equal(sd["linears.8.weight"].view(torch.uint8),
                                    sd["linears.0.weight"].view(torch.uint8)))
        self.assertEqual(tuple(sd["depformer.layers.0.self_attn.in_proj_weight"].shape),
                         (3 * 16 * 16, 16))
        for a, b in zip(step_outputs(loaded, seed=9), step_outputs(online, seed=9)):
            self.assertTrue(torch.equal(a, b))

    def test_file_layout_and_metadata(self):
        header = read_header(os.path.join(self.root, "w16.safetensors"))
        self.assertEqual(header.metadata[METADATA_PREFIX + "version"], "1")
        self.assertEqual(header.metadata[METADATA_PREFIX + "min_features"], str(MIN_FEATURES))
        self.assertEqual(header.tensors["linears.0.weight"].dtype, "F8_E4M3")
        self.assertEqual(header.tensors["linears.0.w8a16_scale"].dtype, "F32")
        self.assertEqual(header.tensors["transformer.layers.0.self_attn.in_proj_weight"].shape, (0,))
        self.assertEqual(header.tensors["transformer.layers.0.self_attn._in_proj_q_weight"].dtype, "F8_E4M3")
        self.assertEqual(header.tensors["emb.0.weight"].dtype, "BF16")
        self.assertEqual(self.summary16["linear_targets"] + self.summary16["in_proj_targets"],
                         len(wq.w8a16_targets(LMModel(device="meta", dtype=BF16, **self.kwargs), MIN_FEATURES)))
        self.assertEqual(len(self.summary16["source_sha256"]), 64)

    def test_source_checksum_is_recorded_and_matches(self):
        from moshi.w8a16_format import sha256_file
        header = read_header(os.path.join(self.root, "w16.safetensors"))
        self.assertEqual(header.metadata[METADATA_PREFIX + "source_sha256"], sha256_file(self.src16))

    def test_verifier_checks_every_tensor_and_catches_a_flipped_byte(self):
        path = os.path.join(self.root, "w16.safetensors")
        rows = wc.verify_converted(self.src16, path, per_kind=None, include_regular=True)
        kinds = {row["kind"] for row in rows}
        self.assertEqual(kinds, {"linear", "gating", "in_proj", "regular"})
        self.assertTrue(all(row["cpu_equal"] for row in rows))
        self.assertEqual(sum(1 for row in rows if row["kind"] != "regular"),
                         self.summary16["linear_targets"] + self.summary16["in_proj_targets"])

        corrupt = os.path.join(self.root, "corrupt.safetensors")
        with open(path, "rb") as handle:
            data = bytearray(handle.read())
        info = read_header(path).tensors["linears.0.weight"]
        start = read_header(path).data_start + info.begin
        data[start + 3] ^= 0x10  # one bit in one stored fp8 weight
        with open(corrupt, "wb") as handle:
            handle.write(bytes(data))
        rows = wc.verify_converted(self.src16, corrupt, per_kind=None)
        bad = [row["tensor"] for row in rows if not row["cpu_equal"]]
        self.assertEqual(bad, ["linears.0.weight"])

    def test_loader_refuses_another_configuration(self):
        other = dict(self.kwargs, dim=64)
        with self.assertRaises(CheckpointFormatError):
            wc.load_lm_w8a16_checkpoint(os.path.join(self.root, "w16.safetensors"), device="cpu", lm_kwargs=other)

    def test_loader_refuses_a_file_from_another_version(self):
        from moshi.w8a16_format import SafetensorsWriter
        path = os.path.join(self.root, "old.safetensors")
        writer = SafetensorsWriter(path, [("x", "U8", (1,), 1)], {METADATA_PREFIX + "version": "0"})
        writer.write("x", b"\x00")
        writer.finish()
        with self.assertRaises(CheckpointFormatError):
            wc.load_lm_w8a16_checkpoint(path, device="cpu", lm_kwargs=self.kwargs)

    def test_converter_refuses_to_overwrite_and_leaves_no_partial_file(self):
        dst = os.path.join(self.root, "again.safetensors")
        wc.convert_checkpoint(self.src16, dst, lm_kwargs=self.kwargs, min_features=MIN_FEATURES, threads=1)
        with self.assertRaises(FileExistsError):
            wc.convert_checkpoint(self.src16, dst, lm_kwargs=self.kwargs, min_features=MIN_FEATURES, threads=1)
        with mock.patch.object(wc, "row_scales", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                wc.convert_checkpoint(self.src16, dst + "2", lm_kwargs=self.kwargs,
                                      min_features=MIN_FEATURES, threads=1)
        self.assertFalse(os.path.exists(dst + "2"))
        self.assertFalse(os.path.exists(dst + "2.partial"))

    def test_converter_refuses_a_source_that_cannot_be_mapped(self):
        bad = dict(self.state16)
        bad["text_linear.weight"] = bad["text_linear.weight"][:, :8].contiguous()
        path = os.path.join(self.root, "bad.safetensors")
        save_file(bad, path)
        from moshi.checkpoint_plan import CheckpointPlanError
        with self.assertRaises(CheckpointPlanError):
            wc.convert_checkpoint(path, os.path.join(self.root, "bad_out.safetensors"),
                                  lm_kwargs=self.kwargs, min_features=MIN_FEATURES, threads=1)


@unittest.skipUnless(HAVE_TORCH and os.environ.get("W8A16_SOURCE") and os.environ.get("W8A16_CONVERTED"),
                     "set W8A16_SOURCE and W8A16_CONVERTED to run the real-tensor check")
class RealTensorSample(unittest.TestCase):
    def test_real_sample_matches_source(self):
        rows = wc.verify_converted(os.environ["W8A16_SOURCE"], os.environ["W8A16_CONVERTED"], per_kind=3)
        kinds = {row["kind"] for row in rows}
        self.assertEqual(kinds, {"linear", "gating", "in_proj"})
        for row in rows:
            self.assertTrue(row["cpu_equal"], row["tensor"])


if __name__ == "__main__":
    unittest.main()
