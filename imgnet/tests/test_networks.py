"""Critic heads, JiT attention/sampling and frozen encoder gradients."""

import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from builders.common import build_representation
from main import setup_distributed
from networks.backbone.jit.denoiser import JiT_models, JiTDenoiser
from networks.backbone.jit.model import Attention, JiT, _RoPECallable
from networks.backbone.jit.modules import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_partial,
    precompute_rope_freqs_2d,
)
from networks.critics import _SpatialTransformerBlockVariant
from networks.representors.siglip2 import SigLIP2Representor
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class CriticHeadTest(unittest.TestCase):
    def test_cosine_attention_is_invariant_to_query_key_rescaling(self):
        torch.manual_seed(93)
        x = torch.randn(2, 4, 8)
        for cosine in (False, True):
            with self.subTest(cosine=cosine):
                block = _SpatialTransformerBlockVariant(
                    8, 2, use_spectral_norm=False, use_cosine_attention=cosine
                )
                rescaled = copy.deepcopy(block)
                with torch.no_grad():
                    rescaled.qkv.weight[:8].mul_(3.0)
                    rescaled.qkv.weight[8:16].mul_(5.0)
                original = block(x)
                changed = rescaled(x)
                if cosine:
                    torch.testing.assert_close(original, changed, rtol=1e-5, atol=1e-6)
                else:
                    self.assertGreater((original - changed).abs().max().item(), 0.01)


class JiTAttentionTest(unittest.TestCase):
    def test_runtime_setup_preserves_math_only_potential_double_backward(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with sdpa_kernel(SDPBackend.MATH), patch.dict(os.environ, {}, clear=True):
            setup_distributed()
            self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
            self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
            self.assertTrue(torch.backends.cuda.math_sdp_enabled())
            model = Attention(32, num_heads=4).to(device).double()
            rope = _RoPECallable(
                precompute_rope_freqs_2d(8, 16), apply_rotary_pos_emb_partial
            )
            query = torch.randn(
                1, 18, 32, device=device, dtype=torch.float64, requires_grad=True
            )
            output = model(query, rope)
            gradient = torch.autograd.grad(
                output.square().mean(), query, create_graph=True
            )[0]
            second = torch.autograd.grad(gradient.square().mean(), query)[0]
            self.assertTrue(torch.isfinite(second).all())
            self.assertGreater(float(second.norm()), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_full_jit_b_bf16_runtime_attention_forward_backward_matches_math(self):
        # Start with all backends permitted, as in a new platform process. Restore
        # flags/environment afterward so other tests do not inherit this policy.
        backends = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        with sdpa_kernel(backends), patch.dict(os.environ, {}, clear=True):
            setup_distributed()
            torch.manual_seed(134)
            model = Attention(768, num_heads=12).cuda()
            frequencies = precompute_rope_freqs_2d(64, 256)
            for prefix in [0, 32]:
                with self.subTest(prefix=prefix):
                    rope = _RoPECallable(
                        frequencies,
                        apply_rotary_pos_emb_partial
                        if prefix
                        else apply_rotary_pos_emb,
                    )
                    x = torch.randn(
                        1, 256 + prefix, 768, device="cuda", requires_grad=True
                    )
                    reference = x.detach().clone().requires_grad_(True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        actual = model(x, rope)
                        gradient = torch.autograd.grad(
                            actual.float().square().mean(), x
                        )[0]
                    with (
                        sdpa_kernel(SDPBackend.MATH),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        expected = model(reference, rope)
                        expected_gradient = torch.autograd.grad(
                            expected.float().square().mean(), reference
                        )[0]
                    self.assertTrue(
                        torch.isfinite(actual).all() and torch.isfinite(gradient).all()
                    )
                    self.assertGreater(float(gradient.norm()), 0)
                    self.assertLess(
                        float(
                            (actual.float() - expected.float()).norm()
                            / expected.float().norm()
                        ),
                        0.02,
                    )
                    self.assertLess(
                        float(
                            (gradient - expected_gradient).norm()
                            / expected_gradient.norm()
                        ),
                        0.05,
                    )


class SamplingTest(unittest.TestCase):
    def model(self, **kwargs):
        def tiny(**kwargs):
            return JiT(hidden_size=32, depth=1, num_heads=4, bottleneck_dim=8,
                       in_context_len=2, in_context_start=0, **kwargs)

        with patch.dict(JiT_models, {"JiT-B": tiny}):
            return JiTDenoiser(input_size=8, patch_size=2, num_classes=2,
                               learned_pe=True, **kwargs).eval()

    def test_potential_gradient_model_generates_finite_images(self):
        torch.manual_seed(173)
        model = self.model(use_potential_grad=True)
        args = SimpleNamespace(num_sampling_steps=2, same_noise=False,
                               sampling_method="euler", interval_min=.1, interval_max=1.)
        initial = torch.randn(2, 3, 8, 8)
        result = model.generate(2, torch.tensor([0, 1]), cfg=1., args=args,
                                verbose=False, z_t=initial)
        self.assertEqual(result.shape, initial.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(result.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # Evaluation wrappers use inference_mode around sample(). The network
        # still needs its internal derivative without accumulating parameter grads.
        with torch.inference_mode():
            inference_result = model.generate(2, torch.tensor([0, 1]), cfg=1., args=args,
                                              verbose=False, z_t=initial)
        torch.testing.assert_close(inference_result, result)
        self.assertFalse(inference_result.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # The public signature defaults args=None; it must provide the same
        # one-step Euler defaults as the backbone adapter.
        args.num_sampling_steps = 1
        explicit = model.generate(2, torch.tensor([0, 1]), cfg=1., args=args,
                                  verbose=False, z_t=initial)
        default = model.generate(2, torch.tensor([0, 1]), cfg=1., verbose=False, z_t=initial)
        torch.testing.assert_close(default, explicit)
        args.num_sampling_steps = 0
        with self.assertRaisesRegex(ValueError, "positive integer"):
            model.generate(2, torch.tensor([0, 1]), cfg=1., args=args,
                           verbose=False, z_t=initial)

    def test_native_noise_generation_respects_input_channels(self):
        torch.manual_seed(174)
        model = self.model(in_channels=4)
        labels = torch.zeros(2, dtype=torch.long)
        for same_noise in (False, True):
            with self.subTest(same_noise=same_noise):
                args = SimpleNamespace(num_sampling_steps=2, same_noise=same_noise,
                                       sampling_method="euler", interval_min=.1, interval_max=1.)
                result = model.generate(2, labels, cfg=1., args=args, verbose=False)
                self.assertEqual(result.shape, (2, 4, 8, 8))
                self.assertTrue(torch.isfinite(result).all())
                if same_noise:
                    torch.testing.assert_close(result[0], result[1])


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.))
        self.config = SimpleNamespace(hidden_size=3, patch_size=2)

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def forward(self, image, output_hidden_states=False):
        tokens = F.avg_pool2d(image, 2).flatten(2).transpose(1, 2) * self.scale
        return SimpleNamespace(last_hidden_state=tokens)


class RepresentorGradientTest(unittest.TestCase):
    def test_pixel_builder_preserves_patches_and_input_gradients(self):
        encoder = build_representation({"arch": "pixel", "patch_size": 2})
        image = torch.arange(96.0).reshape(2, 3, 4, 4).requires_grad_()
        spatial = encoder.extract_featmap(image)
        expected = F.unfold(image, kernel_size=2, stride=2).reshape(2, 12, 2, 2)
        torch.testing.assert_close(spatial, expected)
        spatial.sum().backward()
        torch.testing.assert_close(image.grad, torch.ones_like(image))

    def test_siglip_spatial_features_preserve_requested_input_gradient(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", {"transformers": SimpleNamespace(SiglipVisionModel=Vision)}),
        ):
            encoder = SigLIP2Representor(input_size=4, ckpt_path=directory)
        image = torch.randn(2, 3, 4, 4, requires_grad=True)
        spatial = encoder.extract_featmap(image)
        self.assertTrue(spatial.requires_grad)
        expected = F.avg_pool2d((image - 0.5) / 0.5, 2) * 2
        torch.testing.assert_close(spatial, expected)
        spatial.mean((2, 3)).sum().backward()
        torch.testing.assert_close(image.grad, torch.full_like(image, .25))
        with torch.no_grad():
            self.assertFalse(encoder.extract_featmap(image).requires_grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
