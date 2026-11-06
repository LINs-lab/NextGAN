"""Evaluation entry points and generator-only checkpoint loading."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from builders import build_evaluation_generator
from metrics import eval_all_fds
from metrics.eval import EvaluationGenerator, _build_eval_args, _latest_checkpoint
from metrics.utils import perception_util
from torch import nn

from tests.helpers import public_config


class EvaluationEntryTest(unittest.TestCase):
    def test_inception_cache_hit_loads_weights_without_a_download(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "custom-inception.pth"
            torch.save({"weight": torch.arange(3)}, checkpoint)
            with patch.object(perception_util, "_inception_ckpt_path", return_value=checkpoint), patch.object(
                torch.hub, "load_state_dict_from_url"
            ) as download:
                state = perception_util._load_inception_state()
            torch.testing.assert_close(state["weight"], torch.arange(3))
            download.assert_not_called()

    def test_inception_download_honors_the_selected_checkpoint_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "cache" / "custom-inception.pth"

            def download(url, *, model_dir, file_name, progress, weights_only):
                self.assertEqual(url, perception_util.INCEPTION_URL)
                self.assertTrue(weights_only)
                torch.save({"weight": torch.arange(3)}, Path(model_dir) / file_name)

            with patch.object(perception_util, "_inception_ckpt_path", return_value=checkpoint), patch.object(
                torch.hub, "load_state_dict_from_url", side_effect=download
            ) as fetch:
                state = perception_util._load_inception_state()
            self.assertTrue(checkpoint.is_file())
            torch.testing.assert_close(state["weight"], torch.arange(3))
            fetch.assert_called_once()

    def test_inception_cache_ranks_join_the_same_barrier_for_late_readers(self):
        for preexisting in (False, True):
            with self.subTest(preexisting=preexisting), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "custom-inception.pth"
                expected = {"weight": torch.arange(3)}
                if preexisting:
                    torch.save(expected, checkpoint)
                rank = [0]
                events = []

                def download(url, *, model_dir, file_name, **kwargs):
                    self.assertEqual(rank[0], 0)
                    events.append(("download", rank[0]))
                    torch.save(expected, Path(model_dir) / file_name)

                with (
                    patch.object(perception_util, "_inception_ckpt_path", return_value=checkpoint),
                    patch.object(perception_util, "get_global_rank", side_effect=lambda: rank[0]),
                    patch.object(perception_util, "is_enabled", return_value=True),
                    patch.object(
                        perception_util.dist, "barrier",
                        side_effect=lambda: events.append(("barrier", rank[0])),
                    ),
                    patch.object(torch.hub, "load_state_dict_from_url", side_effect=download) as fetch,
                ):
                    first = perception_util._load_inception_state()
                    self.assertTrue(checkpoint.is_file())
                    rank[0] = 1
                    late = perception_util._load_inception_state()
                expected_events = [] if preexisting else [("download", 0)]
                self.assertEqual(events, expected_events + [("barrier", 0), ("barrier", 1)])
                self.assertEqual(fetch.call_count, 0 if preexisting else 1)
                torch.testing.assert_close(first["weight"], expected["weight"])
                torch.testing.assert_close(late["weight"], expected["weight"])

    def test_inception_float_images_reach_network_in_minus_one_to_one_range(self):
        captured = []

        class CapturedInput(Exception):
            pass

        class Capture(torch.nn.Module):
            def forward(self, images):
                captured.append(images)
                raise CapturedInput

        model = perception_util.InceptionV3()
        model.Conv2d_1a_3x3 = Capture()
        images = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64).view(1, 3, 1, 1)
        with self.assertRaises(CapturedInput):
            model(images.expand(1, 3, 8, 6))
        expected = torch.tensor([-1.0, -0.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 299, 299)
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)

    def test_missing_inception_statistics_fail_before_loading_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [("FID(JiT)", Path(directory) / "jit.npz"),
                     ("FID(ADM)", Path(directory) / "adm.npz")]
            # A directory named like the statistics file is not a usable asset.
            paths[0][1].mkdir()
            with patch.object(eval_all_fds, "INCEPTION_STATS", paths), patch.object(
                eval_all_fds, "load_repr_model"
            ) as load:
                with self.assertRaisesRegex(FileNotFoundError, "DM_FID_STATS_DIR"):
                    eval_all_fds._load_repr_models(["inception"], 256)
                load.assert_not_called()

    def test_inception_keeps_each_available_reference(self):
        real_tensor = torch.tensor
        # Reference selection is independent of GPU feature extraction.
        def cpu_tensor(value, **kwargs):
            return real_tensor(value, dtype=kwargs["dtype"])

        with tempfile.TemporaryDirectory() as directory:
            paths = [("FID(JiT)", Path(directory) / "jit.npz"),
                     ("FID(ADM)", Path(directory) / "adm.npz")]
            model = object()
            for count in (1, 2):
                with self.subTest(reference_count=count):
                    np.savez(paths[count - 1][1], mu=np.zeros(2), sigma=np.eye(2))
                    with patch.object(eval_all_fds, "INCEPTION_STATS", paths), patch.object(
                        eval_all_fds, "load_repr_model", return_value=(model, 2, True, 299)
                    ), patch.object(eval_all_fds.torch, "tensor", side_effect=cpu_tensor):
                        entries = eval_all_fds._load_repr_models(["inception"], 256)
                    self.assertEqual([entry["name"] for entry in entries],
                                     [label for label, _ in paths[:count]])
                    for entry in entries:
                        self.assertIs(entry["model"], model)
                        self.assertEqual(tuple(entry["sigma_ref"].shape), (2, 2))

    def test_latest_checkpoint_ignores_invalid_and_negative_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("tbsm", "logistic_gan", "nextgan"):
                for step in ("10", "2", "100", "invalid", "-1", "backup_999"):
                    (Path(directory) / f"{name}_step_{step}.pt").touch()
                (Path(directory) / f"{name}_step_1000.pt").mkdir()
                selected = _latest_checkpoint(directory, name)
                self.assertEqual(Path(selected).name, f"{name}_step_100.pt")

    def test_invalid_names_alone_do_not_select_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "nextgan_step_invalid.pt").touch()
            (Path(directory) / "nextgan_step_-1.pt").touch()
            self.assertEqual(_latest_checkpoint(directory, "nextgan"), "")

    def test_invalid_model_is_rejected_even_with_explicit_checkpoint(self):
        cfg = {"model": {"name": "gan"}, "evaluation": {"checkpoint": "unused.pt"}}
        with self.assertRaisesRegex(ValueError, "Unknown model.name"):
            _build_eval_args("config.yaml", cfg)

    def test_invalid_weight_selection_fails_before_evaluation_setup(self):
        cfg = {"model": {"name": "nextgan"}, "evaluation": {"weights": "other"}}
        with self.assertRaisesRegex(ValueError, "evaluation.weights"):
            _build_eval_args("config.yaml", cfg)

    def test_metric_labels_identify_the_selected_online_or_ema_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "nextgan_step_2.pt"
            checkpoint.touch()
            for weights in ("online", "ema"):
                cfg = {
                    "model": {"name": "nextgan"},
                    "experiment": {"save_dir": directory},
                    "data": {"resolution": 256, "num_classes": 1000},
                    "evaluation": {"weights": weights, "checkpoint": str(checkpoint)},
                }
                args = _build_eval_args("config.yaml", cfg)
                self.assertEqual(args.weights, weights)


class _EvaluationBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("offset", torch.tensor(0.0))

    def forward(self, x, y, t=None):
        return x * self.weight + self.offset


class EvaluationBuilderTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "checkpoint.pt"
        self.network = patch(
            "builders.evaluation.build_network", side_effect=lambda _: _EvaluationBackbone()
        )
        self.network.start()
        self.addCleanup(self.network.stop)

    def config(self, name="tbsm"):
        cfg = public_config(name)
        cfg["runtime"] = {"device": "cpu", "world_size": 2, "local_rank": 0}
        # Training assets and optimizer settings have no role in inference.
        cfg["model"]["params"]["representations"] = [
            {"arch": "missing", "ckpt_path": "/missing"}
        ]
        cfg["model"]["params"]["trainer"] = {"noise_scale": 0.25}
        return cfg

    def checkpoint(self, prefix="_eval_generator.backbone.", ddp=False):
        online = {"weight": torch.tensor(2.0), "offset": torch.tensor(0.1)}
        ema = {"weight": torch.tensor(3.0), "offset": torch.tensor(-0.2)}
        state = {"generator.backbone." + key: value for key, value in online.items()}
        state.update({prefix + key: value for key, value in ema.items()})
        # Critic states are deliberately incompatible; they must never be loaded.
        state["matchers.0.unused"] = torch.ones(7)
        if ddp:
            state = {
                key.replace("generator.backbone.", "generator.backbone.module.")
                if key.startswith("generator.backbone.")
                else key: value
                for key, value in state.items()
            }
        return {"model": state, "backbone": online, "step": 17}, online, ema

    def test_three_methods_load_distinct_online_and_ema_without_training(self):
        checkpoint, online, ema = self.checkpoint()
        torch.save(checkpoint, self.path)
        with (
            patch(
                "torch.optim.AdamW", side_effect=AssertionError("optimizer allocation")
            ),
            patch("builders.tbsm.build", side_effect=AssertionError("TBSM allocation")),
            patch("builders.gan.build", side_effect=AssertionError("GAN allocation")),
        ):
            for name in ("tbsm", "logistic_gan", "nextgan"):
                cfg = self.config(name)
                original = copy.deepcopy(cfg)
                for weights, expected in (("online", online), ("ema", ema)):
                    with self.subTest(model=name, weights=weights):
                        generator, step = build_evaluation_generator(
                            cfg, self.path, weights
                        )
                        self.assertEqual(step, 17)
                        self.assertFalse(generator.training)
                        self.assertTrue(
                            all(
                                not p.requires_grad and p.grad is None
                                for p in generator.parameters()
                            )
                        )
                        for key, value in expected.items():
                            torch.testing.assert_close(
                                generator.backbone.state_dict()[key],
                                value,
                                rtol=0,
                                atol=0,
                            )
                self.assertEqual(cfg, original)

    def test_current_ddp_checkpoint_loads_online_and_ema(self):
        checkpoint, online, ema = self.checkpoint(ddp=True)
        torch.save(checkpoint, self.path)
        for weights, expected in (("online", online), ("ema", ema)):
            with self.subTest(weights=weights):
                generator, _ = build_evaluation_generator(
                    self.config(), self.path, weights
                )
                for key, value in expected.items():
                    torch.testing.assert_close(
                        generator.backbone.state_dict()[key], value, rtol=0, atol=0
                    )

    def test_old_ema_prefix_and_raw_initialization_are_not_evaluation_formats(self):
        checkpoint, online, _ = self.checkpoint(prefix="_ema_generator.backbone.")
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(ValueError, "no EMA"):
            build_evaluation_generator(self.config(), self.path, "ema")
        torch.save(online, self.path)
        with self.assertRaises(KeyError):
            build_evaluation_generator(self.config(), self.path, "online")

    def test_missing_or_incomplete_ema_is_not_replaced_by_online(self):
        checkpoint, _, _ = self.checkpoint()
        torch.save(
            {"model": {}, "backbone": checkpoint["backbone"], "step": 17}, self.path
        )
        with self.assertRaisesRegex(ValueError, "no EMA"):
            build_evaluation_generator(self.config(), self.path, "ema")
        checkpoint["model"].pop("_eval_generator.backbone.offset")
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            build_evaluation_generator(self.config(), self.path, "ema")

    def test_online_weights_require_an_exact_network_match(self):
        checkpoint, _, _ = self.checkpoint()
        checkpoint["backbone"].pop("offset")
        torch.save(checkpoint, self.path)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            build_evaluation_generator(self.config(), self.path, "online")

    def test_pixel_and_latent_adapter_preserve_noise_scale_and_image_contract(self):
        checkpoint, online, ema = self.checkpoint()
        torch.save(checkpoint, self.path)
        noise = torch.linspace(-0.8, 0.8, 24).reshape(2, 3, 2, 2)
        labels = torch.tensor([0, 1])

        class Decoder(nn.Module):
            def forward(self, x):
                return x * 0.2 + 0.4

        for latent in (False, True):
            cfg = self.config()
            cfg["runtime"].pop("device")
            if latent:
                cfg["latent"] = {"channels": 3, "downsample_ratio": 2}
            original = copy.deepcopy(cfg)
            decoder = Decoder() if latent else None
            with (
                patch("builders.evaluation.build_decoder", return_value=decoder),
                patch("torch.cuda.is_available", return_value=False),
            ):
                for weights, state in (("online", online), ("ema", ema)):
                    adapter = EvaluationGenerator(cfg, str(self.path), weights)
                    actual = adapter.generate(labels, noise=noise)
                    generated = noise * 0.25 * state["weight"] + state["offset"]
                    images = decoder(generated) if latent else generated * 0.5 + 0.5
                    torch.testing.assert_close(
                        actual, images.clamp(0, 1) * 2 - 1, rtol=0, atol=0
                    )
                    self.assertEqual(adapter.current_step, 17)
            self.assertEqual(cfg, original)


if __name__ == "__main__":
    unittest.main()
