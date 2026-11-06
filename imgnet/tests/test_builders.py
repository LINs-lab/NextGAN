"""Configuration, backbone assembly and public method builders."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import torch
from builders.common import (
    build_network,
    public_model_params,
    representation_options,
    training_options,
)
from config import load_config
from methods import TBSM, LogisticGAN, NextGAN
from networks import backbone as backbones
from networks.generator import UniversalAdapter
from torch import nn

from tests.helpers import public_config


class BuilderConfigTest(unittest.TestCase):
    def test_shared_critic_and_representation_overrides(self):
        cfg = public_config("nextgan")
        p = cfg["model"]["params"]
        p["feature_stats_decay"] = 0.95
        p["trainer"]["critic_betas"] = [0.5, 0.9]
        p["representations"].append(
            {
                "arch": "mae_b",
                "input_size": 8,
                "critic": {"hidden_dim": 12},
                "loss_weight": 0.3,
                "feature_stats_decay": 0.98,
            }
        )
        original = copy.deepcopy(cfg)
        a, b = [
            options
            for encoder, options in representation_options(
                public_model_params(cfg), "nextgan"
            )
        ]
        self.assertEqual(cfg, original)
        self.assertEqual(a["critic"]["hidden_dim"], 8)
        self.assertEqual(b["critic"]["hidden_dim"], 12)
        self.assertEqual(b["critic"]["depth"], 1)
        self.assertEqual((a["ema_beta"], b["ema_beta"]), (0.95, 0.98))
        self.assertEqual(b["loss_weight"], 0.3)
        self.assertEqual(
            training_options(p)["critic_betas"], [0.5, 0.9]
        )

    def test_invalid_configs_fail_before_allocating_networks(self):
        from builders import build_training_model

        changes = [
            ("nextgan", {"lambda": 0.9}),
            ("tbsm", {"normalize_v": False}),
            ("tbsm", {"critic": {"hidden_dim": 9, "num_heads": 2}}),
            ("nextgan", {"trainer": {"loss_mode": "logistic"}}),
            ("tbsm", {"representations": [{"arch": "pixel"}]}),
            ("nextgan", {"representations": [{"arch": "missing"}]}),
            ("nextgan", {"representations": [{"arch": "mae_b", "patch_size": 16}]}),
        ]
        with (
            patch("builders.tbsm.build_network") as tbsm_network,
            patch("builders.gan.build_network") as gan_network,
        ):
            for name, change in changes:
                cfg = public_config(name)
                cfg["model"]["params"].update(change)
                with self.assertRaises(ValueError):
                    build_training_model(cfg)
            cfg = public_config()
            cfg["method"] = {}
            with self.assertRaisesRegex(ValueError, "Legacy method"):
                build_training_model(cfg)
            cfg = public_config("missing")
            with self.assertRaisesRegex(ValueError, "Unknown model.name"):
                build_training_model(cfg)
            tbsm_network.assert_not_called()
            gan_network.assert_not_called()

    def test_tbsm_endpoint_keeps_normalized_bearings_and_optional_potential(self):
        cfg = public_config()
        cfg["model"]["params"].update(
            {"lambda": 0.0, "rho": 1.0, "use_potential_grad": True}
        )
        encoder, options = representation_options(public_model_params(cfg), "tbsm")[0]
        self.assertNotIn("normalize_v", options)
        self.assertTrue(options["critic"]["use_potential_grad"])

    def test_defaults_and_input_config_are_preserved(self):
        for name in ("tbsm", "logistic_gan", "nextgan"):
            cfg = public_config(name)
            original = copy.deepcopy(cfg)
            options = training_options(public_model_params(cfg))
            self.assertEqual(options["t_sampling"], [1.0])
            self.assertEqual(options["noise_scale"], 1.0)
            self.assertEqual(cfg, original)
        cfg = public_config()
        options = representation_options(public_model_params(cfg), "tbsm")[0][1]
        self.assertEqual((options["intra_weight"], options["track_weight"]), (0.9, 0.9))
        self.assertFalse(options["critic"]["use_potential_grad"])

    def test_public_schedule_counts_reject_invalid_values_for_every_method(self):
        from builders import build_training_model

        with (
            patch("builders.tbsm.build_network") as tbsm,
            patch("builders.gan.build_network") as gan,
        ):
            for name in ("tbsm", "logistic_gan", "nextgan"):
                for key in ("critic_steps", "grad_accum_steps"):
                    for value in (0, -1, 1.5, True, "2"):
                        with self.subTest(model=name, key=key, value=value):
                            cfg = public_config(name)
                            cfg["model"]["params"]["trainer"][key] = value
                            with self.assertRaisesRegex(ValueError, key):
                                build_training_model(cfg)
            tbsm.assert_not_called()
            gan.assert_not_called()

    def test_all_public_yaml_templates(self):
        import yaml

        root = Path(__file__).resolve().parents[1] / "configs"
        templates = sorted(root.rglob("*.yaml"))
        self.assertEqual(len(templates), 7)
        self.assertEqual(len(list(root.glob("*.yaml"))), 6)
        for path in templates:
            cfg = yaml.safe_load(path.read_text())
            params = public_model_params(cfg)
            representation_options(params, cfg["model"]["name"])
            training_options(params)


class ThreeModelRuntimeTest(unittest.TestCase):
    def test_retained_alternate_critic_architectures_update_generators(self):
        import torch
        from builders import build_training_model
        from torch import nn

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)

            def forward(self, x, y, t=None):
                return self.conv(x)

        class Representor(nn.Module):
            feat_dim = 3

            def extract_featmap(self, image):
                return image

        with (
            patch("builders.gan.build_network", side_effect=lambda _: Backbone()),
            patch(
                "builders.gan.build_representation", side_effect=lambda _: Representor()
            ),
        ):
            for name, arch in (
                ("logistic_gan", "naive"),
                ("nextgan", "naive"),
                ("logistic_gan", "scattering_variant"),
                ("nextgan", "scattering_variant"),
            ):
                with self.subTest(model=name, arch=arch):
                    torch.manual_seed(51)
                    cfg = public_config(name)
                    if arch == "naive":
                        cfg["model"]["params"]["critic"] = {
                            "arch": arch,
                            "use_spectral_norm": True,
                        }
                    else:
                        cfg["model"]["params"]["critic"].update(
                            arch=arch,
                            use_spectral_norm=False,
                            use_cosine_attention=True,
                        )
                    cfg["runtime"] = {"device": torch.device("cpu"), "world_size": 1}
                    model, _ = build_training_model(cfg)
                    before = copy.deepcopy(model.generator.state_dict())
                    for _ in range(2):
                        logs = model.step(
                            lambda: (torch.randn(2, 3, 4, 4), torch.tensor([0, 1]))
                        )
                        self.assertTrue(
                            all(
                                torch.isfinite(torch.as_tensor(v)).all()
                                for v in logs.values()
                            )
                        )
                    self.assertTrue(
                        any(
                            not torch.equal(before[k], v)
                            for k, v in model.generator.state_dict().items()
                        )
                    )

    def test_real_builders_updates_and_generator_only_ema_restore(self):
        import torch
        from builders import build_training_model
        from torch import nn

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)

            def forward(self, x, y, t=None):
                return self.conv(x)

        class Representor(nn.Module):
            feat_dim = 3

            def extract_featmap(self, image):
                return image

        def network(_):
            return Backbone()

        patches = (
            patch("builders.tbsm.build_network", side_effect=network),
            patch("builders.gan.build_network", side_effect=network),
            patch("builders.evaluation.build_network", side_effect=network),
            patch(
                "builders.tbsm.build_representation",
                side_effect=lambda _: Representor(),
            ),
            patch(
                "builders.gan.build_representation", side_effect=lambda _: Representor()
            ),
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            tempfile.TemporaryDirectory() as directory,
        ):
            for name, potential in (
                ("tbsm", False),
                ("tbsm", True),
                ("logistic_gan", False),
                ("nextgan", False),
            ):
                with self.subTest(model=name, potential=potential):
                    torch.manual_seed(7)
                    cfg = public_config(name)
                    if name == "tbsm":
                        cfg["model"]["params"]["use_potential_grad"] = potential
                    cfg["runtime"] = {"device": torch.device("cpu"), "world_size": 1}
                    model, step = build_training_model(cfg)
                    self.assertIs(type(model), {
                        "tbsm": TBSM, "logistic_gan": LogisticGAN,
                        "nextgan": NextGAN,
                    }[name])
                    self.assertEqual(step, 0)
                    self.assertEqual(model.t_sampling, (1.0,))
                    self.assertEqual(model.noise_scale, 1.0)
                    if name == "tbsm":
                        self.assertEqual(
                            model.matchers[0].critic.refine is not None, potential
                        )
                    else:
                        self.assertFalse(hasattr(model, "loss_mode"))
                        self.assertEqual(
                            model.matchers[0].critic.output_mode,
                            "potential",
                        )
                    before = copy.deepcopy(model.generator.state_dict())
                    images = torch.randn(2, 3, 4, 4)
                    labels = torch.tensor([0, 1])
                    logs = (
                        model.step(images, labels)
                        if name == "tbsm"
                        else model.step(lambda: (images, labels))
                    )
                    self.assertTrue(
                        all(
                            torch.isfinite(torch.as_tensor(v)).all()
                            for v in logs.values()
                        )
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(before[k], v)
                            for k, v in model.generator.state_dict().items()
                        )
                    )
                    # Real full checkpoint -> generator-only evaluation path.
                    from builders.common import training_checkpoint
                    state = model.state_dict()
                    path = str(Path(directory) / f"{name}.pt")
                    checkpoint = training_checkpoint(model, 1)
                    self.assertIn("critic_optimizer", checkpoint)
                    self.assertTrue(
                        any(key.startswith("matchers.0.critic.") for key in checkpoint["model"])
                    )
                    torch.save(checkpoint, path)
                    cfg["experiment"]["resume"] = path
                    restored, step = build_training_model(cfg)
                    self.assertEqual(step, 1)
                    for key, value in state.items():
                        torch.testing.assert_close(
                            restored.state_dict()[key], value, rtol=0, atol=0
                        )
                    init_cfg = copy.deepcopy(cfg)
                    init_cfg["experiment"].update(resume="", init_weights=path)
                    initialized, initial_step = build_training_model(init_cfg)
                    self.assertEqual(initial_step, 0)
                    self.assertEqual(
                        initialized.generator_optimizer.state_dict()["state"], {}
                    )
                    eval_cfg = copy.deepcopy(cfg)
                    eval_cfg["model"]["params"]["representations"] = [
                        {"arch": "missing", "ckpt_path": "/missing"}
                    ]
                    from builders import build_evaluation_generator
                    eval_model, step = build_evaluation_generator(eval_cfg, path, "ema")
                    noise = torch.randn_like(images)
                    torch.testing.assert_close(
                        eval_model.to_image(eval_model(noise, labels, torch.ones(2))).float().clamp(0, 1),
                        model.sample(labels, noise),
                        rtol=0,
                        atol=0,
                    )

                    from metrics.eval import EvaluationGenerator

                    raw_cfg = copy.deepcopy(eval_cfg)
                    raw_cfg["runtime"].pop("device")
                    original = copy.deepcopy(raw_cfg)
                    for weights in ("ema", "online"):
                        # This oracle compares against the CPU training model.
                        # Keep it on CPU even on a CUDA-equipped test machine.
                        with patch("torch.cuda.is_available", return_value=False):
                            adapter = EvaluationGenerator(raw_cfg, path, weights)
                        self.assertEqual(adapter.current_step, 1)
                        actual = adapter.generate(labels, noise=noise)
                        expected = (
                            model.sample(labels, noise)
                            if weights == "ema"
                            else model.generator.to_image(
                                model.generator(noise, labels, torch.ones(2))
                            )
                            .float()
                            .clamp(0, 1)
                        )
                        torch.testing.assert_close(
                            actual, expected * 2 - 1, rtol=0, atol=0
                        )
                    self.assertEqual(raw_cfg, original)


class ConfigUtilsTest(unittest.TestCase):
    def test_expands_paths_and_preserves_integer_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "runtime:\n  nproc_per_node: ${TEST_NPROC}\n"
                "  deterministic: $TEST_FLAG\n"
                "  choices: ${TEST_CHOICES}\n"
                "data:\n  data_dir: ${TEST_ROOT}/latents\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"TEST_NPROC": "4", "TEST_ROOT": "/shared/data",
                 "TEST_FLAG": "false", "TEST_CHOICES": "[0.5, 1.0]"},
                clear=False,
            ):
                cfg = load_config(path)
        self.assertEqual(cfg["runtime"]["nproc_per_node"], 4)
        self.assertIs(cfg["runtime"]["deterministic"], False)
        self.assertEqual(cfg["runtime"]["choices"], [0.5, 1.0])
        self.assertEqual(cfg["data"]["data_dir"], "/shared/data/latents")

    def test_unset_variable_fails_with_clear_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("value: ${DEFINITELY_UNSET_NEXTGAN_VAR}\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(KeyError, "DEFINITELY_UNSET_NEXTGAN_VAR"):
                    load_config(path)

    def test_template_representation_paths_expand_model_zoo(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        weights = {
            "mae_b": "mae/vit_base_patch16_224.mae.pth",
            "siglip2_b": "siglip2/siglip2-base-patch16-256",
            "resnet18": "resnet/resnet18-f37072fd.pth",
        }
        with mock.patch.dict(os.environ, {"DM_MODEL_ZOO": "/local/model cache"}):
            for path in root.glob("*.yaml"):
                cfg = load_config(path)
                for rep in cfg["model"]["params"]["representations"]:
                    self.assertEqual(
                        rep["ckpt_path"],
                        str(Path("/local/model cache/pretrained") / weights[rep["arch"]]),
                    )


class AssemblyTest(unittest.TestCase):
    def config(self):
        return {
            "network": {"name": "jit_b", "params": {"input_size": 8}},
            "data": {"num_classes": 2},
        }

    def test_jit_and_pixeldit_do_not_require_dit_dependencies(self):
        code = """
import sys, importlib.abc
sys.path[:] = PATHS
class BlockTimm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'timm' or fullname.startswith('timm.'):
            raise ModuleNotFoundError('timm intentionally unavailable')
sys.meta_path.insert(0, BlockTimm())
import networks.backbone.jit
import networks.backbone.pixeldit
import builders.common
assert 'networks.backbone.dit' not in sys.modules
""".replace("PATHS", json.dumps(sys.path))
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_adapter_conversion_does_not_mutate_config(self):
        class Backbone(nn.Module):
            def forward(self, x, y, t=None):
                return torch.ones_like(x) * 2

        adapter = {
            "source_prediction_type": "velocity",
            "target_prediction_type": "x",
        }
        cfg = self.config()
        cfg["network"]["adapter"] = adapter
        original = copy.deepcopy(cfg)
        with patch.object(backbones, "jit_b", return_value=Backbone()) as factory:
            model = build_network(cfg)
        factory.assert_called_once_with(num_classes=2, input_size=8)
        self.assertIsInstance(model, UniversalAdapter)
        x = torch.zeros(2, 3, 8, 8)
        time = torch.full((2,), 0.4)
        expected = x - time[:, None, None, None] * 2
        torch.testing.assert_close(model(x, torch.tensor([0, 1]), time), expected)
        self.assertEqual(cfg, original)

    def test_old_nested_adapter_is_rejected_before_constructing_backbone(self):
        for value in (None, {}, {"source_prediction_type": "x"}):
            with self.subTest(value=value):
                cfg = self.config()
                cfg["network"]["params"]["adapter"] = value
                with patch.object(backbones, "jit_b") as factory:
                    with self.assertRaisesRegex(
                        ValueError, "removed; use network.adapter"
                    ):
                        build_network(cfg)
                    factory.assert_not_called()

    def test_invalid_adapter_mapping_is_rejected_before_construction(self):
        cfg = self.config()
        cfg["network"]["adapter"] = True
        with patch.object(backbones, "jit_b") as factory:
            with self.assertRaisesRegex(TypeError, "network.adapter"):
                build_network(cfg)
            factory.assert_not_called()

    def test_dit_adapter_supports_omitted_time(self):
        from networks.backbone.dit.adapter import DiT_models, DiTBackboneAdapter
        from networks.backbone.dit.model import DiT

        with patch.dict(
            DiT_models,
            {
                "tiny": lambda **kwargs: DiT(
                    input_size=8,
                    patch_size=2,
                    in_channels=4,
                    hidden_size=32,
                    depth=1,
                    num_heads=4,
                    **kwargs,
                )
            },
        ):
            model = DiTBackboneAdapter("tiny", num_classes=2).eval()
        x = torch.randn(2, 4, 8, 8)
        y = torch.tensor([0, 1])
        torch.testing.assert_close(model(x, y), model(x, y, t=x.new_zeros(2)))


if __name__ == "__main__":
    unittest.main()
