"""LogisticGAN objectives; fresh-sample scheduling lives in GANBase."""

import torch.nn.functional as F
from ._gan_base import GANBase, _generator_critic


class LogisticGAN(GANBase):
    """Non-saturating logistic GAN with critic-first generator updates."""

    def _critic_loss(self, features, labels, time):
        loss = features[0][0].new_tensor(0.0)
        logs = {}
        for index, (matcher, (real_feat, fake_feat)) in enumerate(
            zip(self.matchers, features)
        ):
            real_score = matcher.critic(labels, real_feat, time)
            fake_score = matcher.critic(labels, fake_feat.detach(), time)
            loss = (
                loss
                + (F.softplus(-real_score).mean() + F.softplus(fake_score).mean())
                * matcher.loss_weight
            )
            suffix = f"_{index}" if len(features) > 1 else ""
            logs[f"real_energy{suffix}"] = real_score.mean().detach()
            logs[f"fake_energy{suffix}"] = fake_score.mean().detach()
        return loss, logs, None

    def _generator_loss(self, features, labels, time, context):
        loss = features[0][0].new_tensor(0.0)
        for matcher, (_, fake_feat) in zip(self.matchers, features):
            score = _generator_critic(matcher)(labels, fake_feat, time)
            loss = loss + F.softplus(-score).mean() * matcher.loss_weight
        return loss
