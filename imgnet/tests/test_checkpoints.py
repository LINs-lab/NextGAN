"""Initialization candidates, training checkpoints and resumed schedules."""

import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from builders import build_evaluation_generator, build_training_model
from builders.common import (
    load_backbone_state,
    load_checkpoint,
    training_checkpoint,
    validate_checkpoint_path,
)
from torch import nn

from tests.helpers import public_config


class _TrainingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x, y, t=None):
        return self.conv(x)


class _TrainingRepresentor(nn.Module):
    feat_dim = 3

    def extract_featmap(self, image):
        return image


class CheckpointAccumulationTest(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        for name in ("tbsm", "gan", "evaluation"):
            self.contexts.enter_context(
                patch(
                    f"builders.{name}.build_network", side_effect=lambda _: _TrainingBackbone()
                )
            )
            if name == "evaluation":
                continue
            self.contexts.enter_context(
                patch(
                    f"builders.{name}.build_representation",
                    side_effect=lambda _: _TrainingRepresentor(),
                )
            )
        self.directory = self.contexts.enter_context(tempfile.TemporaryDirectory())

    def config(self, name="nextgan", accumulation=2, critics=3):
        cfg = public_config(name)
        cfg["runtime"] = {"device": torch.device("cpu"), "world_size": 1}
        cfg["model"]["params"]["trainer"].update(
            grad_accum_steps=accumulation, critic_steps=critics
        )
        return cfg

    def assert_exact(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_exact(x, y)
        else:
            self.assertEqual(a, b)

    def test_tbsm_pending_gradients_resume_matches_uninterrupted_updates(self):
        labels = torch.tensor([0, 1])
        for accumulation, critics, cut in ((2, 3, 1), (2, 3, 5), (3, 2, 1)):
            with self.subTest(accumulation=accumulation, critics=critics, cut=cut):
                torch.manual_seed(39)
                cfg = self.config("tbsm", accumulation, critics)
                model, _ = build_training_model(cfg)
                batches = [torch.randn(2, 3, 2, 2) for _ in range(cut + 4)]
                for x in batches[:cut]:
                    model.step(x, labels)
                checkpoint = training_checkpoint(model, cut)
                self.assertEqual(checkpoint["training_state"]["version"], 1)
                self.assertTrue(checkpoint["training_state"]["gradients"])
                path = str(Path(self.directory) / "state.pt")
                torch.save(checkpoint, path)
                rng = torch.get_rng_state()
                expected_logs = [model.step(x, labels) for x in batches[cut:]]
                expected = copy.deepcopy(training_checkpoint(model, cut + 4))

                cfg["experiment"]["resume"] = path
                restored, step = build_training_model(cfg)
                self.assertEqual(step, cut)
                actual_gradients = training_checkpoint(restored, cut)["training_state"][
                    "gradients"
                ]
                self.assert_exact(
                    actual_gradients, checkpoint["training_state"]["gradients"]
                )
                torch.set_rng_state(rng)
                actual_logs = [restored.step(x, labels) for x in batches[cut:]]
                self.assert_exact(actual_logs, expected_logs)
                self.assert_exact(training_checkpoint(restored, cut + 4), expected)

    def test_gan_cycle_boundary_resume_matches_uninterrupted_updates(self):
        labels = torch.tensor([0, 1])
        for name in ("logistic_gan", "nextgan"):
            for accumulation, critics, cut in ((2, 3, 1), (3, 2, 2), (1, 1, 1)):
                with self.subTest(
                    model=name, accumulation=accumulation, critics=critics, cut=cut
                ):
                    torch.manual_seed(39)
                    cfg = self.config(name, accumulation, critics)
                    model, _ = build_training_model(cfg)
                    per_cycle = accumulation * critics
                    batches = [
                        torch.randn(2, 3, 2, 2) for _ in range((cut + 2) * per_cycle)
                    ]
                    stream = iter(batches)

                    def next_batch():
                        return next(stream), labels

                    for _ in range(cut):
                        model.step(next_batch)
                    checkpoint = training_checkpoint(model, cut)
                    self.assertEqual(checkpoint["training_state"]["version"], 2)
                    self.assertEqual(checkpoint["training_state"]["gradients"], {})
                    path = str(Path(self.directory) / "gan-cycle.pt")
                    torch.save(checkpoint, path)
                    rng = torch.get_rng_state()
                    expected_logs = [model.step(next_batch) for _ in range(2)]
                    expected = copy.deepcopy(training_checkpoint(model, cut + 2))

                    cfg["experiment"]["resume"] = path
                    restored, step = build_training_model(cfg)
                    self.assertEqual(step, cut)
                    self.assertEqual(
                        training_checkpoint(restored, cut)["training_state"][
                            "gradients"
                        ],
                        {},
                    )
                    # Production checkpoints do not save RNG or this data cursor.
                    stream = iter(batches[cut * per_cycle :])
                    torch.set_rng_state(rng)
                    actual_logs = [restored.step(next_batch) for _ in range(2)]
                    self.assert_exact(actual_logs, expected_logs)
                    self.assert_exact(training_checkpoint(restored, cut + 2), expected)

    def test_invalid_gradient_or_schedule_fails_before_loading_weights(self):
        cfg = self.config("tbsm")
        model, _ = build_training_model(cfg)
        model.step(torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))
        state = copy.deepcopy(training_checkpoint(model, 1))
        key = next(iter(state["training_state"]["gradients"]))
        bad_states = []
        for field, value in (
            ("grad_accum_steps", 4),
            ("critic_steps", 1),
            ("version", 2),
        ):
            bad = copy.deepcopy(state)
            bad["training_state"][field] = value
            bad_states.append(bad)
        for gradients in (
            {"missing.parameter": torch.ones(1)},
            {key: torch.ones(7)},
            {
                key: torch.ones_like(
                    state["training_state"]["gradients"][key], dtype=torch.float64
                )
            },
        ):
            bad = copy.deepcopy(state)
            bad["training_state"]["gradients"] = gradients
            bad_states.append(bad)
        for bad in bad_states:
            fresh, _ = build_training_model(cfg)
            before = copy.deepcopy(fresh.state_dict())
            path = str(Path(self.directory) / "bad.pt")
            torch.save(bad, path)
            resume_cfg = copy.deepcopy(cfg)
            resume_cfg["experiment"]["resume"] = path
            with self.assertRaises(ValueError):
                load_checkpoint(fresh, resume_cfg)
            self.assert_exact(fresh.state_dict(), before)

    def test_gan_rejects_previous_schedule_checkpoint_before_loading_weights(self):
        for name in ("logistic_gan", "nextgan"):
            with self.subTest(model=name):
                cfg = self.config(name)
                model, _ = build_training_model(cfg)
                state = training_checkpoint(model, 0)
                self.assertEqual(state["training_state"]["version"], 2)
                state["training_state"]["version"] = 1
                path = str(Path(self.directory) / "old-gan-schedule.pt")
                torch.save(state, path)
                fresh, _ = build_training_model(cfg)
                before = copy.deepcopy(fresh.state_dict())
                cfg["experiment"]["resume"] = path
                with self.assertRaisesRegex(ValueError, "training_state"):
                    load_checkpoint(fresh, cfg)
                self.assert_exact(fresh.state_dict(), before)

    def test_resume_requires_training_state_even_at_an_accumulation_boundary(self):
        for step in (1, 2):
            cfg = self.config("tbsm")
            model, _ = build_training_model(cfg)
            for _ in range(step):
                model.step(torch.randn(2, 3, 2, 2), torch.tensor([0, 1]))
            state = training_checkpoint(model, step)
            state.pop("training_state")
            path = str(Path(self.directory) / "missing-state.pt")
            torch.save(state, path)
            cfg["experiment"]["resume"] = path
            with self.assertRaisesRegex(ValueError, "training_state"):
                build_training_model(cfg)

    def test_gan_rejects_pending_gradients_in_a_completed_cycle_checkpoint(self):
        cfg = self.config("nextgan")
        model, _ = build_training_model(cfg)
        checkpoint = training_checkpoint(model, 0)
        name, parameter = next(
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        )
        checkpoint["training_state"]["gradients"] = {name: torch.ones_like(parameter)}
        path = str(Path(self.directory) / "invalid-pending.pt")
        torch.save(checkpoint, path)
        before = copy.deepcopy(model.state_dict())
        cfg["experiment"]["resume"] = path
        with self.assertRaisesRegex(ValueError, "completed generator update"):
            load_checkpoint(model, cfg)
        self.assert_exact(model.state_dict(), before)

    def test_training_requires_optimizer_state_but_evaluation_does_not(self):
        cfg = self.config(accumulation=1, critics=1)
        model, _ = build_training_model(cfg)
        model.step(lambda: (torch.randn(2, 3, 2, 2), torch.tensor([0, 1])))
        state = training_checkpoint(model, 1)
        for key in ("generator_optimizer", "critic_optimizer"):
            bad = copy.deepcopy(state)
            bad.pop(key)
            path = str(Path(self.directory) / "missing-optimizer.pt")
            torch.save(bad, path)
            cfg["experiment"]["resume"] = path
            with self.assertRaisesRegex(ValueError, key):
                build_training_model(cfg)
            _, step = build_evaluation_generator(
                cfg, cfg["experiment"]["resume"], "ema"
            )
            self.assertEqual(step, 1)
            cfg["experiment"]["resume"] = ""


class _CandidateBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., 2.]))
        self.model = nn.Module()
        self.model.weight = nn.Parameter(torch.tensor([3., 4.]))


class CandidateTest(unittest.TestCase):
    def test_only_the_selected_candidate_changes_weights(self):
        model = _CandidateBackbone()
        missing = model.model.weight.detach().clone()
        state = {"weight": torch.tensor([9., 8.])}
        result = load_backbone_state(model, state)
        self.assertEqual(result.missing_keys, ["model.weight"])
        torch.testing.assert_close(model.weight, state["weight"])
        torch.testing.assert_close(model.model.weight, missing)

    def test_unrelated_checkpoint_fails_without_changing_weights(self):
        model = _CandidateBackbone()
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "compatible"):
            load_backbone_state(model, {"unrelated": torch.ones(2)})
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_shape_mismatch_is_rejected_before_partial_load(self):
        model = _CandidateBackbone()
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "compatible"):
            load_backbone_state(model, {"weight": torch.tensor([9., 8.]),
                                        "model.weight": torch.ones(3)})
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])


class CheckpointPathTest(unittest.TestCase):
    def test_missing_selected_path_fails_before_building_for_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.pt")
            for name in ("tbsm", "logistic_gan", "nextgan"):
                for key in ("resume", "init_weights"):
                    for evaluation in (False, True):
                        with self.subTest(model=name, key=key, evaluation=evaluation):
                            cfg = public_config(name)
                            cfg["experiment"][key] = missing
                            with (
                                patch("builders.tbsm.build_network") as tbsm,
                                patch("builders.gan.build_network") as gan,
                            ):
                                with self.assertRaisesRegex(
                                    FileNotFoundError, "checkpoint"
                                ):
                                    if evaluation:
                                        build_evaluation_generator(cfg, missing)
                                    else:
                                        build_training_model(cfg)
                                tbsm.assert_not_called()
                                gan.assert_not_called()

    def test_existing_resume_takes_precedence_over_unused_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            path.touch()
            cfg = public_config()
            cfg["experiment"].update(
                resume=str(path), init_weights=str(path.parent / "unused.pt")
            )
            validate_checkpoint_path(cfg)

    def test_empty_paths_allow_fresh_training(self):
        validate_checkpoint_path(public_config())


if __name__ == "__main__":
    unittest.main()
