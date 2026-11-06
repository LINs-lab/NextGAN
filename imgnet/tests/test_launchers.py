"""Training and evaluation shell launch behavior."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN_RECORDER = '''import json, os, sys
from pathlib import Path
Path(os.environ["TEST_LAUNCH_LOG"]).write_text(json.dumps({
    "argv": sys.argv, "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "distributed": {key: os.environ.get(key) for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]}}))
raise SystemExit(int(os.environ.get("TEST_EXIT_CODE", "0")))
'''


class TrainLauncherTest(unittest.TestCase):
    def launch(self, nproc=1, deterministic=False, *, exit_code=0, optimize=False, scheduler_environment=False, prepare_images=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'bin').mkdir()
            shutil.copy2(ROOT / 'scripts/train.sh', root / 'scripts/train.sh')
            shutil.copy2(ROOT / 'config.py', root / 'config.py')
            (root / 'main.py').write_text(TRAIN_RECORDER)
            torchrun = root / 'bin/torchrun'
            torchrun.write_text(f'#!{sys.executable}\n' + TRAIN_RECORDER)
            torchrun.chmod(0o755)
            config = root / 'config with spaces.yaml'
            cfg = {'runtime': {'nproc_per_node': nproc, 'deterministic': deterministic},
                   'data': {'name': 'in1k_latent'}}
            if prepare_images:
                from PIL import Image
                (root / 'builders').mkdir()
                (root / 'builders/__init__.py').touch()
                shutil.copy2(ROOT / 'builders/datasets.py', root / 'builders/datasets.py')
                images = root / 'images/class0'
                images.mkdir(parents=True)
                Image.new('RGB', (4, 4)).save(images / 'sample.png')
                cfg['data'] = {'name': 'in1k', 'data_dir': str(root / 'images'),
                               'cache_dir': str(root / 'index'), 'num_classes': 1, 'resolution': 256}
            config.write_text(yaml.safe_dump(cfg))
            log = root / 'calls.json'
            env = dict(os.environ, DM_MODEL_ZOO=str(root / 'cache'),
                       TEST_LAUNCH_LOG=str(log), TEST_EXIT_CODE=str(exit_code),
                       TEST_TRAIN_PROCESSES='2', CUBLAS_WORKSPACE_CONFIG='inherited')
            env['PATH'] = os.pathsep.join([str(root / 'bin'), str(Path(sys.executable).parent), env['PATH']])
            env['PYTHONPATH'] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p)
            # This test owns PATH and CUBLAS; image shell hooks must not replace them.
            env.pop('BASH_ENV', None)
            env.pop('ENV', None)
            env = {key: value for key, value in env.items() if not key.startswith('BASH_FUNC_')}
            if scheduler_environment:
                env.update(RANK='0', WORLD_SIZE='1', MASTER_ADDR='scheduler-master', MASTER_PORT='12345')
                env.pop('LOCAL_RANK', None)
            if optimize:
                env['PYTHONOPTIMIZE'] = '1'
            result = subprocess.run(['bash', 'scripts/train.sh', str(config)],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=30)
            return result, json.loads(log.read_text()) if log.exists() else None

    def test_single_process_preserves_config_argument_and_existing_backend_setting(self):
        result, call = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(call['argv'][1], '--config')
        self.assertEqual(Path(call['argv'][2]).name, 'config with spaces.yaml')
        self.assertEqual(call['cublas'], 'inherited')

    def test_single_process_does_not_join_inherited_scheduler_process_group(self):
        result, call = self.launch(scheduler_environment=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(call['distributed'], {
            'RANK': None, 'WORLD_SIZE': None, 'LOCAL_RANK': None,
            'MASTER_ADDR': None, 'MASTER_PORT': None,
        })

    def test_multiple_processes_expand_typed_environment_and_disable_restarts(self):
        result, call = self.launch('${TEST_TRAIN_PROCESSES}', True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(call['argv'][1:5], ['--standalone', '--nproc_per_node=2',
                                         '--max-restarts=0', 'main.py'])
        self.assertEqual(call['cublas'], ':4096:8')

    def test_invalid_runtime_never_launches_even_when_python_is_optimized(self):
        for nproc in (True, 0, -1, 1.5, '2'):
            with self.subTest(nproc=nproc):
                result, call = self.launch(nproc, optimize=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(call)
                self.assertIn('runtime.nproc_per_node', result.stderr)
        for deterministic in (1, 'true', None):
            with self.subTest(deterministic=deterministic):
                result, call = self.launch(deterministic=deterministic, optimize=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(call)
                self.assertIn('runtime.deterministic', result.stderr)

    def test_cold_rgb_index_is_ready_before_distributed_launch(self):
        result, call = self.launch(nproc=2, prepare_images=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ImageNet index ready: 1 images", result.stderr)
        self.assertEqual(call['argv'][1], '--standalone')

    def test_training_failure_is_returned_by_both_routes(self):
        for nproc in (1, 2):
            with self.subTest(nproc=nproc):
                result, call = self.launch(nproc, exit_code=17)
                self.assertIsNotNone(call)
                self.assertEqual(result.returncode, 17)


EVAL_RECORDER = '''import json, os, sys
from pathlib import Path
Path(os.environ["TEST_LAUNCH_LOG"]).write_text(json.dumps(sys.argv[1:]))
raise SystemExit(int(os.environ["TEST_EXIT_CODE"]))
'''


class EvalLauncherTest(unittest.TestCase):
    def launch(self, *, eval_nproc=None, exit_code=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            torchrun = binary_dir / "torchrun"
            torchrun.write_text(f"#!{sys.executable}\n" + EVAL_RECORDER)
            torchrun.chmod(0o755)
            log = root / "calls.json"
            config = root / "config with spaces.yaml"
            config.write_text("{}\n")
            checkpoint = root / "checkpoint with spaces.pt"
            environment = {
                key: value for key, value in os.environ.items()
                if key not in ("BASH_ENV", "ENV", "EVAL_NPROC_PER_NODE")
                and not key.startswith("BASH_FUNC_")
            }
            environment.update(
                DM_MODEL_ZOO=str(root / "models"),
                DM_DATA_ZOO=str(root / "data"),
                NPROC_PER_NODE="9",
                TEST_LAUNCH_LOG=str(log),
                TEST_EXIT_CODE=str(exit_code),
            )
            environment["PATH"] = str(binary_dir) + os.pathsep + environment["PATH"]
            if eval_nproc is not None:
                environment["EVAL_NPROC_PER_NODE"] = str(eval_nproc)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/eval.sh"), str(config),
                 "--checkpoint", str(checkpoint)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result, json.loads(log.read_text())

    def test_default_ignores_old_process_count_alias(self):
        result, arguments = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            arguments[:5],
            ["--standalone", "--nproc_per_node=1", "--max-restarts=0", "-m", "metrics.eval"],
        )
        self.assertEqual(Path(arguments[5]).name, "config with spaces.yaml")
        self.assertEqual(arguments[6], "--checkpoint")
        self.assertEqual(Path(arguments[7]).name, "checkpoint with spaces.pt")

    def test_explicit_evaluation_process_count(self):
        result, arguments = self.launch(eval_nproc=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(arguments[1], "--nproc_per_node=3")

    def test_evaluation_failure_code_is_preserved(self):
        result, arguments = self.launch(eval_nproc=2, exit_code=17)
        self.assertEqual(arguments[1], "--nproc_per_node=2")
        self.assertEqual(result.returncode, 17)


if __name__ == "__main__":
    unittest.main()
