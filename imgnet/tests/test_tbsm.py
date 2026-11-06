"""TBSM vector targets, optimizer schedules and latent decoding."""

import copy
import importlib.util
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from methods.tbsm import TBSM, RepresentationScattering
from networks.generator import Generator
from torch import nn


class _VectorRepresentor(nn.Module):
    feat_dim = 2

    def extract_featmap(self, image):
        return image


class TBSMVectorTest(unittest.TestCase):
    def test_bearings_and_mixture_match_independent_formula(self):
        real = torch.tensor([[[3.0, 4.0]]])
        fake = torch.zeros_like(real, requires_grad=True)
        source = torch.tensor([[[0.0, 2.0]]])
        tracked = torch.tensor([[[0.3, -0.7]]], requires_grad=True)
        for lam, rho in ((0.9, 0.9), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
            with self.subTest(lam=lam, rho=rho):
                matcher = RepresentationScattering(_VectorRepresentor(), lam, rho)
                target, vector = matcher.compute_three_body_scattering_target(
                    real, fake, source, tracked
                )
                expected = (
                    real / (real.norm() + 1e-6) - lam * source / (source.norm() + 1e-6)
                ) * 2**0.5
                torch.testing.assert_close(vector, expected)
                torch.testing.assert_close(
                    target, rho * tracked.detach() + (1 - rho) * expected
                )
                self.assertFalse(target.requires_grad)

    def test_zero_bearings_are_finite(self):
        x = torch.zeros(2, 3, 2)
        matcher = RepresentationScattering(_VectorRepresentor(), 0.9, 0.9)
        target, vector = matcher.compute_three_body_scattering_target(x, x, x)
        torch.testing.assert_close(target, x)
        torch.testing.assert_close(vector, x)


class _ScheduleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 3, 1)

    def forward(self, x, y, t):
        return self.backbone(x)

    def to_image(self, x):
        return x


class _ScheduleMatcher(nn.Module):
    lam = 0

    def __init__(self):
        super().__init__()
        self.critic = nn.Conv2d(3, 3, 1)

    def init_critic(self, n):
        pass

    def compute_loss(self, real, fake, source, y, t):
        v = self.critic(fake.detach())
        target = (real - fake.detach()).detach()
        lg = (fake - (fake.detach() + v).detach()).square().mean()
        lv = (v - target).square().mean()
        return lg, lv, {"generator": lg.detach(), "critic": lv.detach()}


def _schedule_model(**kw):
    return TBSM(_ScheduleGenerator(), [_ScheduleMatcher()], 2, 1e-3, 1e-3, 0.9, [1.0], **kw)


class CriticStepsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_frequency_and_no_generator_update_on_critic_only_steps(self):
        for k, a in [(1, 1), (3, 1), (3, 2)]:
            m = _schedule_model(critic_steps=k, grad_accum_steps=a)
            real = torch.randn(2, 3, 2, 2)
            y = torch.tensor([0, 1])
            with (
                patch.object(
                    m.generator_optimizer, "step", wraps=m.generator_optimizer.step
                ) as gs,
                patch.object(
                    m.critic_optimizer, "step", wraps=m.critic_optimizer.step
                ) as ts,
                patch.object(m, "_ema_update", wraps=m._ema_update) as es,
            ):
                for i in range(2 * k * a):
                    before = [p.detach().clone() for p in m.generator.parameters()]
                    m.step(real, y)
                    self.assertEqual(ts.call_count, (i + 1) // a)
                    self.assertEqual(gs.call_count, (i + 1) // (k * a))
                    self.assertEqual(es.call_count, gs.call_count)
                    if (i + 1) % (k * a):
                        self.assertTrue(
                            all(
                                torch.equal(x, p)
                                for x, p in zip(before, m.generator.parameters())
                            )
                        )
                        if (i // a) % k != k - 1:
                            self.assertTrue(
                                all(p.grad is None for p in m.generator.parameters())
                            )
                self.assertEqual(ts.call_count, 2 * k)
                self.assertEqual(gs.call_count, 2)

    def test_resume_phase_with_accumulation(self):
        m = _schedule_model(critic_steps=3, grad_accum_steps=2)
        m.set_training_step(4)
        with (
            patch.object(
                m.generator_optimizer, "step", wraps=m.generator_optimizer.step
            ) as gs,
            patch.object(
                m.critic_optimizer, "step", wraps=m.critic_optimizer.step
            ) as ts,
        ):
            for _ in range(2):
                m.step(torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))
            self.assertEqual(ts.call_count, 1)
            self.assertEqual(gs.call_count, 1)

    def test_invalid_or_conflicting_schedule(self):
        for k in [0, -1, 1.5, True, "2"]:
            with self.assertRaises(ValueError):
                _schedule_model(critic_steps=k)
        with self.assertRaises(TypeError):
            _schedule_model(critic_updates_per_step=1)


ROOT = Path(__file__).resolve().parents[1]


class _LatentRepresentor(nn.Module):
    feat_dim = 3

    def extract_featmap(self, x):
        return x


class _LatentBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 4, 1)
        self.register_buffer("rotary_cache", torch.ones(2, 2))

    def forward(self, x, y, t):
        return self.projection(x)


class _LatentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 3, 1)
        self.requires_grad_(False).eval()

    def forward(self, x):
        return self.projection(x).sigmoid()


class LatentTests(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        torch.set_num_threads(1)
        torch.manual_seed(213)

    def test_decoder_dispatch_defaults_and_invalid_type(self):
        from builders.common import build_decoder

        sd, va = Mock(), Mock()
        self.contexts.enter_context(patch("builders.common.SDVAE", sd))
        self.contexts.enter_context(patch("builders.common.VAVAE", va))
        build = build_decoder
        self.assertIsNone(build(None, "cpu"))
        build(dict(decoder_path="sd"), "cpu")
        sd.assert_called_once_with(model_path="sd", multiplier=0.18215)
        build(dict(decoder_type="vavae", decoder_path="va", stat_path="stats"), "cpu")
        va.assert_called_once_with(model_path="va", stat_path="stats", multiplier=1.0)
        with self.assertRaises(ValueError):
            build(dict(decoder_type="wrong", decoder_path="x"), "cpu")
        with self.assertRaises(KeyError):
            build(dict(decoder_type="vavae", decoder_path="va"), "cpu")

    def test_actual_vavae_denormalization_preserves_input_gradient(self):
        spec = importlib.util.spec_from_file_location(
            "tested_vavae", ROOT / "networks/vaes/vavae.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw_seen = []

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.2))

            def decode(self, raw):
                raw_seen.append(raw.detach().clone())
                return types.SimpleNamespace(sample=raw[:, :3] * self.scale)

            def encode(self, image):
                return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mean=image))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.pt"
            mean, std = torch.full((1, 4, 1, 1), 0.3), torch.full((1, 4, 1, 1), 2.0)
            torch.save(dict(mean=mean, std=std), path)
            fake_diffusers = types.ModuleType("diffusers.models")
            fake_diffusers.AutoencoderKL = types.SimpleNamespace(
                from_pretrained=lambda p: Autoencoder()
            )
            with patch.dict("sys.modules", {"diffusers.models": fake_diffusers}):
                decoder = module.VAVAE("unused", str(path), multiplier=2.0)
            latent = torch.zeros(2, 4, 2, 2, requires_grad=True)
            image = decoder(latent)
            torch.testing.assert_close(raw_seen[0], latent.detach() / 2 * std + mean)
            image.sum().backward()
            torch.testing.assert_close(
                latent.grad[:, :3], torch.full_like(latent.grad[:, :3], 0.1)
            )
            self.assertTrue(
                all(
                    p.grad is None and not p.requires_grad for p in decoder.parameters()
                )
            )
            decoder.train()
            self.assertFalse(decoder.training)
            source = torch.full_like(latent, 0.6)
            torch.testing.assert_close(
                decoder.encode(source), (source * 2 - 1 - mean) / std * 2
            )

    def test_sdvae_scaling_clamp_and_frozen_input_gradient(self):
        from networks.vaes import SDVAE

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.2))

            def decode(self, raw):
                return types.SimpleNamespace(sample=raw * self.scale)

        fake_diffusers = types.ModuleType("diffusers.models")
        fake_diffusers.AutoencoderKL = types.SimpleNamespace(
            from_pretrained=lambda _: Autoencoder()
        )
        with patch.dict("sys.modules", {"diffusers.models": fake_diffusers}):
            decoder = SDVAE("unused", multiplier=0.5)
        latent = torch.zeros(1, 4, 2, 2, requires_grad=True)
        image = decoder(latent)
        torch.testing.assert_close(image, torch.full_like(image, 0.5))
        image.sum().backward()
        torch.testing.assert_close(latent.grad, torch.full_like(latent, 0.2))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in decoder.parameters()))
        decoder.train()
        self.assertFalse(decoder.training)
        extreme = torch.tensor([-10.0, 10.0]).view(1, 1, 1, 2)
        torch.testing.assert_close(decoder(extreme), torch.tensor([0.0, 1.0]).view_as(extreme))

    def test_available_native_arms_update_latent_generator_and_restore_decoded_ema(
        self,
    ):
        arms = ["vector", "potential"]
        for arm in arms:
            with self.subTest(arm=arm):
                decoder = _LatentDecoder()
                scattering = RepresentationScattering(
                    _LatentRepresentor(),
                    0.9,
                    0.9,
                    critic=dict(
                        hidden_dim=16,
                        time_dim=8,
                        depth=1,
                        num_heads=2,
                        use_potential_grad=arm == "potential",
                    ),
                )
                model = TBSM(
                    Generator(_LatentBackbone(), decoder),
                    [scattering],
                    2,
                    1e-4,
                    1e-3,
                    0.999,
                    [1.0],
                    critic_steps=1,
                )
                ema_params = dict(model._eval_generator.named_parameters())
                for name, param in model.generator.named_parameters():
                    torch.testing.assert_close(
                        ema_params[name], param, rtol=0, atol=0
                    )
                initial = copy.deepcopy(model.generator.backbone.state_dict())
                fixed = copy.deepcopy(decoder.state_dict())
                seen = []

                def hook(module, args, output):
                    if output.requires_grad:
                        output.register_hook(lambda g: seen.append(g.detach().clone()))

                handle = decoder.register_forward_hook(hook)
                for _ in range(3):
                    real, y = torch.randn(2, 4, 2, 2), torch.tensor([0, 1])
                    logs = model.step(real, y)
                    self.assertTrue(
                        all(
                            torch.isfinite(torch.as_tensor(v)).all()
                            for v in logs.values()
                        )
                    )
                handle.remove()
                self.assertTrue(any(g.abs().sum() > 0 for g in seen))
                self.assertTrue(all(torch.isfinite(g).all() for g in seen))
                self.assertTrue(
                    any(
                        not torch.equal(v, initial[n])
                        for n, v in model.generator.backbone.state_dict().items()
                    )
                )
                self.assertTrue(
                    all(
                        torch.equal(v, fixed[n])
                        for n, v in decoder.state_dict().items()
                    )
                )
                self.assertTrue(all(p.grad is None for p in decoder.parameters()))
                self.assertIs(model._eval_generator.decoder, decoder)
                self.assertFalse(any("decoder" in n for n in model.state_dict()))
                noise = torch.randn(2, 4, 2, 2)
                expected = model.sample(y, noise)
                saved = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    for p in model.generator.parameters():
                        p.add_(1.0)
                model.load_state_dict(saved, strict=True)
                torch.testing.assert_close(
                    model.sample(y, noise), expected, rtol=0, atol=0
                )
                self.assertEqual(expected.shape, (2, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
