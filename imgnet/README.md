# NextGAN on ImageNet

Training and evaluation for class-conditional, one-step image generation.
Run the commands below from `imgnet/`.

[Project overview](../README.md) · [Paper](https://arxiv.org/abs/2611.06270) · [Asset preparation](ASSETS.md)

## Included DiT recipe

[`configs/reproduced/nextgan_dit_vavae_256.yaml`](configs/reproduced/nextgan_dit_vavae_256.yaml)
records the existing DiT-XL/1 + VA-VAE experiment configuration:

| Setting | Value |
| --- | --- |
| Dataset | ImageNet-1K, 256 × 256 |
| Generator | DiT-XL/1, pretrained initialization |
| Decoder | VA-VAE, 32 latent channels, 16× downsampling |
| Training representations | MAE-B/16, SigLIP2-B/16, ResNet18 |
| Generator updates | 100,000 |
| Critic updates per generator update | 2 |
| Training batch size | 32 per GPU × 8 GPUs; 256 per critic update |
| Generator / critic learning rate | `1e-5` / `5e-5` |
| Generator EMA | `0.999` |
| Class dropout | `0.0` |
| Training seed | 42 |

### Recorded recipe results

The existing experiment records report the following FID values for this
configuration:

| Generator / decoder | Best FID (ADM) ↓ | IS at best-FID checkpoint ↑ | Best checkpoint | FID (ADM) at 100K ↓ |
| --- | --- | --- | --- | --- |
| DiT-XL/1 + VA-VAE | 1.27 | 258.9 | 95K | 1.30 |

These measurements use EMA weights, 50,000 generated images, seed 1, eight GPU
ranks, batch size 256 per GPU, BF16, one generator evaluation and `cfg=1`.
The best checkpoint was selected from 11 evaluations at 50K–100K in 5K
increments, using one training seed.

The same 95K checkpoint has FID(JiT) 1.20. All 11 evaluations and
checkpoint hashes are preserved in the
[checkpoint sweep CSV](../assets/readme/imagenet-reproduction.csv).

### Relationship to the paper

The recorded FID(ADM) values above have not yet been reconciled with the
[paper results](../README.md#imagenet-results) at the checkpoint and evaluation
protocol level. The paper's main table does not label its FID as `FID(ADM)` or
`FID(JiT)`; compare values only after matching checkpoints and evaluators.

The current code uses unit-weight field regression, with target displacement
`real_features - fake_features`. [Appendix A of the paper](https://arxiv.org/html/2611.06270v1#A1)
uses a regression scale of λ=5 and a target displacement divided by λ. The
included configuration documents the current implementation; matching the
paper's exact experimental settings requires aligning this loss scaling.

## Prepare the environment and assets

Training and evaluation require Linux, CUDA, Bash and Python 3.10–3.12.
The reproduction entry installs the pinned packages in
[`requirements.txt`](requirements.txt) into `.venv-reproduce/`.

```bash
export DM_MODEL_ZOO=/path/to/model_zoo
export DM_DATA_ZOO=/path/to/data_zoo
```

Edit the `/path/to/` values in the included DiT configuration. Supply:

1. ImageNet latent shards with `latents`, `latents_flip` and `labels`.
2. Pretrained DiT-XL/1 weights matching the 32-channel VA-VAE latent space.
3. The matching converted Diffusers VA-VAE model and `mean` / `std` statistics.
4. MAE, SigLIP2 and ResNet18 weights.
5. Inception weights and `guided_diffusion_stats.npz` for the reported FID(ADM).

The expected training representation and evaluation weight layout is:

```text
model_zoo/
├── pretrained/
│   ├── mae/vit_base_patch16_224.mae.pth
│   ├── resnet/resnet18-f37072fd.pth
│   └── siglip2/siglip2-base-patch16-256/
└── hub/checkpoints/weights-inception-2015-12-05-6726825d.pth

data_zoo/
└── fid_stats/guided_diffusion_stats.npz
```

[ASSETS.md](ASSETS.md) specifies the formats and latent scaling conventions.
Public checkpoint downloads and a versioned asset acquisition/conversion
manifest are pending. Obtain the actual assets before running these commands.

## Train and evaluate the included recipe

Check the configured assets first:

```bash
python3 reproduce.py \
  --config configs/reproduced/nextgan_dit_vavae_256.yaml \
  --stage check \
  --runs-dir ./reproductions/nextgan-dit-256
```

Then train to 100K generator updates and evaluate the final checkpoint:

```bash
EVAL_NPROC_PER_NODE=8 python3 reproduce.py \
  --config configs/reproduced/nextgan_dit_vavae_256.yaml \
  --skip-install \
  --runs-dir ./reproductions/nextgan-dit-256
```

Always pass the NextGAN configuration explicitly with `--config`. The first
command sets up the environment and checks assets; `--skip-install` on later
commands verifies the existing pinned environment. Use a new `--runs-dir` for
each fresh experiment.

The run directory contains `reproduction*.json` records and a
`nextgan_dit_vavae_256/` subdirectory with the resolved `config.yaml`, training
checkpoints, logs and evaluation summaries. Reproduction records include source,
configuration and model-asset hashes; full dataset contents are not hashed.

To inspect the planned commands without installing, writing files or training:

```bash
python3 reproduce.py \
  --config configs/reproduced/nextgan_dit_vavae_256.yaml \
  --dry-run
```

## Evaluate saved checkpoints

For individual evaluations, install the pinned environment and run:

```bash
source .venv-reproduce/bin/activate
EVAL_NPROC_PER_NODE=8 bash scripts/eval.sh \
  configs/reproduced/nextgan_dit_vavae_256.yaml \
  --checkpoint /path/to/training/nextgan_step_95000.pt
```

Use the asset-edited configuration above. The evaluator loads the generator's
EMA weights and decoder without constructing training representations, critics
or optimizers. Evaluation alone does not require ImageNet training shards or
the training representation weights.

The included configuration leaves `evaluation.save_dir` unset, so direct
evaluations write to `<experiment.save_dir>/eval/<checkpoint stem>/`.
Metrics are saved in `final_eval_summary.csv` in that directory. Re-evaluating
the same checkpoint replaces that directory by default; choose a new
`evaluation.save_dir` to keep multiple evaluations.

To sweep the checkpoints used for best-checkpoint selection:

```bash
for step in $(seq 50000 5000 100000); do
  EVAL_NPROC_PER_NODE=8 bash scripts/eval.sh \
    configs/reproduced/nextgan_dit_vavae_256.yaml \
    --checkpoint /path/to/training/nextgan_step_${step}.pt
done
```

Keep the evaluation seed, batch size and rank count fixed across checkpoints:
changing batch size or GPU count changes the sampled noise bank. `FID(ADM)`
uses `guided_diffusion_stats.npz`; `FID(JiT)` uses `jit_in256_stats.npz`.
Compare results with the same metric label and reference statistics. The
reproduction entry evaluates the final checkpoint; selecting the best FID
requires the checkpoint sweep.

## Resume training

```bash
EVAL_NPROC_PER_NODE=8 python3 reproduce.py \
  --config configs/reproduced/nextgan_dit_vavae_256.yaml \
  --resume \
  --skip-install \
  --runs-dir ./reproductions/nextgan-dit-256
```

Resume restores the model, critic, optimizers, EMA and update counters from the
latest numeric checkpoint. Keep the critic and accumulation schedules fixed.
Checkpoints do not restore RNG state or the data-loader cursor, so resumed
training is not bitwise equivalent to an uninterrupted run.

To evaluate the latest checkpoint of an existing reproduction without training,
use `--stage eval` with the same `--config` and `--runs-dir`. Each invocation
preserves previous reproduction evaluation outputs in a separate directory.

## Other NextGAN configurations

| Configuration | Generator and input space | Role |
| --- | --- | --- |
| [`nextgan_pixel_256.yaml`](configs/nextgan_pixel_256.yaml) | JiT-B, RGB pixels | General starting template |
| [`nextgan_latent_256.yaml`](configs/nextgan_latent_256.yaml) | DiT-XL/2 + SD-VAE | General starting template |
| [`nextgan_dit_vavae_256.yaml`](configs/reproduced/nextgan_dit_vavae_256.yaml) | DiT-XL/1 + VA-VAE | Included experiment recipe |

Templates require matching pretrained weights and data. The SD-VAE and VA-VAE
configurations use different channels, spatial scales and normalization.

For manual training in the activated pinned environment:

```bash
bash scripts/train.sh configs/nextgan_pixel_256.yaml
```

Set the asset paths before running. `runtime.nproc_per_node` controls the
single-node training launcher; `EVAL_NPROC_PER_NODE` controls the evaluation
launcher separately. A different GPU count or batch size defines a different
training configuration.

## Implementation notes

NextGAN samples an independent interpolation coefficient for each sample and
representation branch. Each generator update follows `critic_steps` critic
updates with fresh microbatches; the generator queries the updated critic at
fake endpoints. See [`methods/nextgan.py`](methods/nextgan.py) and
[`methods/_gan_base.py`](methods/_gan_base.py).

For component and runtime checks in the installed environment:

```bash
python3 -m unittest discover -s tests
```

These checks cover execution and controlled numerical behavior; actual
pretrained assets and full ImageNet metrics require separate validation.

[Project overview](../README.md) · [Asset preparation](ASSETS.md) · [License](../LICENSE)
