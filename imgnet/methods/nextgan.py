"""NextGAN potential-gradient regression and endpoint generator objective."""

import torch
from ._gan_base import GANBase, _field_regression_loss, _generator_critic


class NextGAN(GANBase):
    """Fit a potential gradient; G maximizes the fake-endpoint potential."""

    def _critic_loss(self, features, labels, time):
        loss = features[0][0].new_tensor(0.0)
        logs = {}
        for index, (matcher, (real_feat, fake_feat)) in enumerate(
            zip(self.matchers, features)
        ):
            alpha = torch.rand(
                real_feat.shape[0], 1, 1, device=real_feat.device, dtype=real_feat.dtype
            )
            displacement = real_feat - fake_feat.detach()
            query = (
                (alpha * real_feat + (1.0 - alpha) * fake_feat.detach())
                .detach()
                .requires_grad_(True)
            )
            kwargs = (
                {"needs_double_backward": True}
                if matcher.critic_arch == "scattering_variant"
                else {}
            )
            potential = matcher.critic(labels, query, time, **kwargs)
            field = torch.autograd.grad(potential.sum(), query, create_graph=True)[0]
            field_loss, residual = _field_regression_loss(field, displacement)
            loss = loss + field_loss * matcher.loss_weight
            suffix = f"_{index}" if len(features) > 1 else ""
            logs[f"field_mse{suffix}"] = field_loss.detach()
            logs[f"field_rms{suffix}"] = field.detach().square().mean().sqrt()
            logs[f"field_target_rms{suffix}"] = (
                displacement.detach().square().mean().sqrt()
            )
            logs[f"field_residual_rms{suffix}"] = (
                residual.detach().square().mean().sqrt()
            )
        return loss, logs, None

    def _generator_loss(self, features, labels, time, context):
        loss = features[0][0].new_tensor(0.0)
        for matcher, (_, fake_feat) in zip(self.matchers, features):
            potential = _generator_critic(matcher)(labels, fake_feat, time)
            loss = loss - potential.mean() * matcher.loss_weight
        return loss
