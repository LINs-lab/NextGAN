"""GAN losses, fresh samples and independent accumulation oracles."""

import unittest
from unittest.mock import patch

import torch
from main import next_training_batch
from methods._gan_base import _field_regression_loss, _freeze_parameters
from methods.logistic_gan import LogisticGAN
from methods.nextgan import NextGAN
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from tests.helpers import (
    _Critic,
    _fresh_real,
    _Generator,
    _RecordingSGD,
)


class FieldMseAccumulationTest(unittest.TestCase):
    CASES = ((1, 1), (1, 2), (2, 1), (3, 2), (5, 2), (3, 5), (2, 5))

    def test_interpolation_is_sampled_independently_for_each_representation(self):
        critics = [_Critic(), _Critic()]
        model = NextGAN(
            _Generator(), critics, gen_lr=0.001, critic_lr=0.002,
            ema_decay=0.9, t_sampling=[1.0],
        )
        queries = []
        handles = [critic.critic.register_forward_pre_hook(
            lambda module, args: queries.append(args[1].detach().clone())
        ) for critic in critics]
        real = torch.ones(2, 1, 2, dtype=torch.float64)
        fake = torch.zeros_like(real)
        alphas = [real.new_tensor([0.2, 0.7]).view(2, 1, 1),
                  real.new_tensor([0.8, 0.3]).view(2, 1, 1)]
        try:
            with patch("methods.nextgan.torch.rand", side_effect=alphas):
                loss, _, _ = model._critic_loss(
                    [(real, fake), (real, fake)], torch.tensor([0, 1]),
                    torch.ones(2, dtype=torch.float64),
                )
            for query, alpha in zip(queries, alphas):
                torch.testing.assert_close(query, alpha.expand_as(real))
            self.assertEqual(len(queries), 2)
            loss.backward()
            self.assertTrue(all(torch.isfinite(c.critic.weight.grad).all() for c in critics))
        finally:
            for handle in handles:
                handle.remove()

    def _check_schedule(self, accumulation, critic_steps, resume_steps=None):
        generator = _Generator()
        critic = _Critic()
        gan = NextGAN(
            generator=generator,
            critics=[critic],
            gen_lr=0.02,
            critic_lr=0.05,
            ema_decay=0.9,
            t_sampling=[1.0],
            noise_scale=0.0,
            grad_accum_steps=accumulation,
            critic_steps=critic_steps,
        )
        events = []
        d_optimizer = _RecordingSGD(critic.critic.weight, 0.05, events, "D")
        g_optimizer = _RecordingSGD(generator.backbone.bias, 0.02, events, "G")
        gan.critic_optimizer = d_optimizer
        gan.generator_optimizer = g_optimizer
        reference_d = d_optimizer.parameter.detach().clone()
        reference_g = g_optimizer.parameter.detach().clone()
        reference_ema = reference_g.clone()
        consumed = []

        def next_batch():
            real = _fresh_real(len(consumed))
            consumed.append(real)
            return real, torch.tensor([0, 1])

        for cycle in range(3):
            if cycle == 1 and resume_steps is not None:
                # The saved outer step must not shift a microbatch schedule.
                gan.set_training_step(resume_steps)
            expected_d_gradients = []
            for critic_step in range(critic_steps):
                start = (cycle * critic_steps + critic_step) * accumulation
                joined_real = torch.cat(
                    [_fresh_real(start + micro) for micro in range(accumulation)]
                )
                joined_fake = reference_g.view(1, 1, -1).expand_as(joined_real)
                weight = reference_d.detach().requires_grad_(True)
                displacement = joined_real - joined_fake
                loss = 0.5 * (weight - displacement).square().flatten(1).sum(1).mean()
                gradient = torch.autograd.grad(loss, weight)[0]
                expected_d_gradients.append(gradient)
                reference_d = (weight - 0.05 * gradient).detach()

            # Independent endpoint potential on the final concatenated group,
            # evaluated with the critic after its last complete update.
            bias = reference_g.detach().requires_grad_(True)
            live_fake = bias.view(1, 1, -1).expand_as(joined_real)
            generator_loss = -(live_fake * reference_d).flatten(1).sum(1).mean()
            expected_g_gradient = torch.autograd.grad(generator_loss, bias)[0]
            reference_g = (bias - 0.02 * expected_g_gradient).detach()
            reference_ema = 0.9 * reference_ema + 0.1 * reference_g

            gan.step(next_batch)

            self.assertEqual(len(consumed), (cycle + 1) * critic_steps * accumulation)
            self.assertEqual(events, (["D"] * critic_steps + ["G"]) * (cycle + 1))
            self.assertEqual(len(g_optimizer.gradients), cycle + 1)
            for offset, gradient in enumerate(expected_d_gradients):
                torch.testing.assert_close(
                    d_optimizer.gradients[cycle * critic_steps + offset],
                    gradient,
                    rtol=1e-12,
                    atol=1e-12,
                )
            torch.testing.assert_close(
                g_optimizer.gradients[-1], expected_g_gradient, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                d_optimizer.parameter, reference_d, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                g_optimizer.parameter, reference_g, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                gan._eval_generator.backbone.bias, reference_ema, rtol=1e-12, atol=1e-12
            )
            self.assertIsNone(d_optimizer.parameter.grad)
            self.assertIsNone(g_optimizer.parameter.grad)

        expected_batches = 3 * critic_steps * accumulation
        self.assertEqual(len(critic.feature_calls), 2 * expected_batches)
        self.assertEqual(len(critic.normalization_calls), 2 * expected_batches)
        self.assertEqual(sum(critic.normalization_calls), expected_batches)
        # Only the last accumulation group in each cycle needs a live G graph.
        self.assertEqual(
            sum(grad for _, grad in critic.feature_calls), 3 * accumulation
        )
        for actual, expected in zip(
            consumed, map(_fresh_real, range(expected_batches))
        ):
            torch.testing.assert_close(actual, expected)

    def test_field_mse_accumulated_fresh_batches_match_concatenation(self):
        for accumulation, critics in self.CASES:
            with self.subTest(accumulation=accumulation, critics=critics):
                self._check_schedule(accumulation, critics)

    def test_completed_generator_cycle_can_resume_training_step(self):
        self._check_schedule(3, 2, resume_steps=35)


class Witness(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_mode = "potential"
        self.weight = nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))

    def forward(self, labels, tokens, time):
        return (tokens * self.weight).flatten(1).sum(1)


class Critic(_Critic):
    def __init__(self):
        super().__init__()
        self.critic = Witness()


class AccumulationTest(unittest.TestCase):
    def test_gradients_and_updates_match_fresh_concatenated_batches(self):
        for cls in (LogisticGAN,):
            for accumulation, critics in (
                (1, 1),
                (1, 2),
                (2, 1),
                (3, 2),
                (5, 2),
                (3, 5),
                (2, 5),
            ):
                with self.subTest(
                    model=cls.__name__, accumulation=accumulation, critics=critics
                ):
                    self.check_case(cls, accumulation, critics)

    def check_case(self, cls, accumulation, critics):
        generator = _Generator()
        critic = Critic()
        model = cls(
            generator,
            [critic],
            gen_lr=0.02,
            critic_lr=0.05,
            ema_decay=0.9,
            t_sampling=[1.0],
            noise_scale=0,
            grad_accum_steps=accumulation,
            critic_steps=critics,
        )
        events = []
        d = _RecordingSGD(critic.critic.weight, 0.05, events, "D")
        g = _RecordingSGD(generator.backbone.bias, 0.02, events, "G")
        model.critic_optimizer = d
        model.generator_optimizer = g
        expected_d = d.parameter.detach().clone()
        expected_g = g.parameter.detach().clone()
        expected_ema = expected_g.clone()
        consumed = []

        def next_batch():
            real = _fresh_real(len(consumed))
            consumed.append(real)
            return real, torch.tensor([0, 1])

        for cycle in range(3):
            expected_gradients = []
            for critic_step in range(critics):
                start = (cycle * critics + critic_step) * accumulation
                joined_real = torch.cat(
                    [_fresh_real(start + micro) for micro in range(accumulation)]
                )
                joined_fake = expected_g.view(1, 1, -1).expand_as(joined_real)
                old_weight = expected_d.detach().clone()
                weight = old_weight.requires_grad_(True)
                real_score = (joined_real * weight).flatten(1).sum(1)
                fake_score = (joined_fake * weight).flatten(1).sum(1)
                loss = (
                    F.softplus(-real_score).mean() + F.softplus(fake_score).mean()
                )
                gradient = torch.autograd.grad(loss, weight)[0]
                expected_gradients.append(gradient)
                expected_d = (weight - 0.05 * gradient).detach()

            bias = expected_g.detach().requires_grad_(True)
            live_fake = bias.view(1, 1, -1).expand_as(joined_real)
            # Logistic GAN uses the final NEW critic for its G loss.
            generator_loss = F.softplus(
                -(live_fake * expected_d).flatten(1).sum(1)
            ).mean()
            expected_g_gradient = torch.autograd.grad(generator_loss, bias)[0]
            expected_g = (bias - 0.02 * expected_g_gradient).detach()
            expected_ema = 0.9 * expected_ema + 0.1 * expected_g

            model.step(next_batch)

            self.assertEqual(len(consumed), (cycle + 1) * critics * accumulation)
            cycle_events = ["D"] * critics + ["G"]
            self.assertEqual(events, cycle_events * (cycle + 1))
            self.assertEqual(len(d.gradients), (cycle + 1) * critics)
            self.assertEqual(len(g.gradients), cycle + 1)
            for offset, gradient in enumerate(expected_gradients):
                torch.testing.assert_close(
                    d.gradients[cycle * critics + offset],
                    gradient,
                    rtol=1e-12,
                    atol=1e-12,
                )
            torch.testing.assert_close(
                g.gradients[-1], expected_g_gradient, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(d.parameter, expected_d, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(g.parameter, expected_g, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(
                model._eval_generator.backbone.bias,
                expected_ema,
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertIsNone(g.parameter.grad)
            self.assertIsNone(d.parameter.grad)

        expected_batches = 3 * critics * accumulation
        self.assertEqual(len(critic.feature_calls), 2 * expected_batches)
        self.assertEqual(len(critic.normalization_calls), 2 * expected_batches)
        self.assertEqual(sum(critic.normalization_calls), expected_batches)
        self.assertEqual(
            sum(grad for _, grad in critic.feature_calls), 3 * accumulation
        )


class RecordingSampler(DistributedSampler):
    def __init__(self, dataset):
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=False)
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        super().set_epoch(epoch)


class GANDataStreamTest(unittest.TestCase):
    def test_fresh_batches_continue_across_epochs_inside_generator_updates(self):
        dataset = TensorDataset(
            torch.arange(12, dtype=torch.float64).reshape(6, 1, 2) / 10,
            torch.arange(6),
        )
        sampler = RecordingSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
        data_iter, epoch = iter(dataloader), 0
        seen = []

        def next_batch():
            nonlocal data_iter, epoch
            x, y, data_iter, epoch = next_training_batch(
                data_iter, dataloader, sampler, epoch
            )
            seen.extend(y.tolist())
            return x, y

        model = NextGAN(
            _Generator(),
            [_Critic()],
            gen_lr=0.001,
            critic_lr=0.002,
            ema_decay=0.9,
            t_sampling=[1.0],
            critic_steps=2,
            grad_accum_steps=2,
        )
        for _ in range(2):
            logs = model.step(next_batch)
            self.assertTrue(all(torch.isfinite(v).all() for v in logs.values()))
        self.assertEqual(seen, list(range(6)) * 2 + list(range(4)))
        self.assertEqual(sampler.epochs, [1, 2])
        self.assertEqual(model._completed_steps, 2)


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, y, t):
        return x * self.scale


class _DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _DummyBackbone()

    def forward(self, x, y, t):
        return self.backbone(x, y, t)

    def to_image(self, x):
        return x


class _DummyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, y, tokens, t, **kwargs):
        return tokens.flatten(1).mean(dim=1) * self.scale


class _RepresentationCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = _DummyCritic()
        self.critic_arch = "scattering"
        self.loss_weight = 1.0

    def _feature_tokens(self, image, requires_grad):
        return image

    def _normalize(self, tokens, update_stats):
        return tokens


def _make_gan():
    return NextGAN(
        generator=_DummyGenerator(),
        critics=[_RepresentationCritic()],
        gen_lr=1e-3,
        critic_lr=1e-3,
        ema_decay=0.9,
        t_sampling=[1.0],
    )


class GanLossHelperTest(unittest.TestCase):
    def test_resume_keeps_distinct_ema_and_online_weights(self):
        original = _make_gan()
        with torch.no_grad():
            original.generator.backbone.scale.fill_(2.0)
            original._eval_generator.backbone.scale.fill_(1.25)
        restored = _make_gan()
        restored.load_state_dict(original.state_dict(), strict=True)
        torch.testing.assert_close(restored.generator.backbone.scale, torch.tensor(2.0))
        torch.testing.assert_close(
            restored._eval_generator.backbone.scale, torch.tensor(1.25)
        )

    def test_field_mse_sums_features_then_averages_batch(self):
        field = torch.zeros(2, 3, 4)
        displacement = torch.ones_like(field)
        loss, _ = _field_regression_loss(field, displacement)
        torch.testing.assert_close(loss, torch.tensor(6.0))

    def test_field_mse_matches_dual_value_and_critic_gradient(self):
        # Simpson integration is exact here: a quadratic potential has a
        # linear field along a segment and a quadratic squared field energy.
        real = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64)
        fake = torch.tensor([[-2.0, 0.5], [2.0, -1.0]], dtype=torch.float64)
        displacement = real - fake
        curvature = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
        slope = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)

        def potential(x):
            return (0.5 * curvature * x.square() + slope * x).sum(1)

        regression = real.new_zeros(())
        energy = real.new_zeros(())
        for alpha, mass in ((0.0, 1 / 6), (0.5, 4 / 6), (1.0, 1 / 6)):
            query = ((1 - alpha) * fake + alpha * real).requires_grad_(True)
            field = torch.autograd.grad(
                potential(query).sum(), query, create_graph=True
            )[0]
            loss, _ = _field_regression_loss(field, displacement)
            regression = regression + mass * loss
            energy = energy + mass * 0.5 * field.square().sum(1).mean()
        dual = potential(fake).mean() - potential(real).mean() + energy
        constant = displacement.square().sum(1).mean() / 2
        torch.testing.assert_close(regression, dual + constant)
        regression_grad = torch.autograd.grad(
            regression, (curvature, slope), retain_graph=True
        )
        dual_grad = torch.autograd.grad(dual, (curvature, slope))
        for actual, expected in zip(regression_grad, dual_grad):
            torch.testing.assert_close(actual, expected)

    def test_field_mse_generator_gradient_is_endpoint_potential_only(self):
        gan = _make_gan()
        real = torch.ones(2, 3, 4)
        fake = torch.zeros(2, 3, 4, requires_grad=True)
        labels = torch.zeros(2, dtype=torch.long)
        before = gan.matchers[0].critic.scale.detach().clone()
        for _ in range(2):
            critic_loss, _, _ = gan._critic_loss([(real, fake)], labels, torch.ones(2))
            critic_loss.backward()
            self.assertIsNone(fake.grad)
            gan.critic_optimizer.step()
            gan.critic_optimizer.zero_grad(set_to_none=True)
        with _freeze_parameters(gan.matchers):
            gan._generator_loss([(real, fake)], labels, torch.ones(2), None).backward()
        after = gan.matchers[0].critic.scale.detach()
        self.assertFalse(torch.equal(before, after))
        # The critic's field-MSE backward must not leak through fake endpoints.
        expected = torch.ones_like(fake) * (-after / fake.numel())
        torch.testing.assert_close(fake.grad, expected)
        self.assertIsNone(gan.matchers[0].critic.scale.grad)

    def test_field_regression_zero_at_target(self):
        displacement = torch.tensor([[[2.0, -4.0]], [[6.0, 8.0]]])
        field = displacement.clone()
        loss, residual = _field_regression_loss(field, displacement)
        torch.testing.assert_close(loss, torch.tensor(0.0))
        torch.testing.assert_close(residual, torch.zeros_like(field))

    def test_field_mse_has_unit_displacement_target_without_lambda(self):
        gan = _make_gan()
        real = torch.randn(2, 3, 4)
        fake = torch.randn(2, 3, 4, requires_grad=True)
        labels = torch.zeros(2, dtype=torch.long)
        _, logs, _ = gan._critic_loss([(real, fake)], labels, torch.ones(2))
        expected_rms = (real - fake.detach()).square().mean().sqrt()
        torch.testing.assert_close(logs["field_target_rms"], expected_rms)


if __name__ == "__main__":
    unittest.main()
