# ImageNet asset contracts

Run from `imgnet/`. Replace the template paths with your local assets. Model
weights and datasets are not included. Select only the assets needed by your
configuration; generator-only evaluation does not load training representations
or construct training algorithms, critics or optimizers. Current training checkpoints
provide `backbone` for online evaluation and `model._eval_generator.backbone.*`
for EMA, with strict network matching. Old training checkpoint layouts are not
supported. Pretrained initialization assets use the separate initialization loader.
The critic optimizer is stored as `critic_optimizer`, with critic network weights
under `model.matchers.<index>.critic.*`.

## Training assets

| Configuration field | Required contents |
| --- | --- |
| `data.data_dir`, `data.name: in1k` | ImageFolder class directories, either directly below the path or below its `train/` directory. Class ordering follows ImageFolder. |
| `data.data_dir`, `data.name: in1k_latent` | `.safetensors` shards with `latents` and `latents_flip` shaped `[N,C,H,W]`, and integer `labels` shaped `[N]`. Both latent arrays must use the same encoder and scale. |
| `experiment.init_weights` | Weights matching `network.name`, architecture parameters and any `network.adapter` conventions. Training starts at step zero with fresh optimizers. |
| `experiment.resume` | A training checkpoint with optimizers and a matching schedule: GAN version 2 at a complete G update, or TBSM version 1 including pending gradients. Takes precedence over initialization. |
| Representation `mae_b` / `ckpt_path` | A state dictionary matching timm `vit_base_patch16_224.mae` with `num_classes=0`; `.safetensors` or a PyTorch checkpoint. The loader checks keys strictly. A generic MAE pretraining checkpoint may require conversion. |
| Representation `siglip2_b` / `ckpt_path` | A local Transformers model directory for `google/siglip2-base-patch16-256`, containing configuration and weights. |
| Representation `resnet18` / `ckpt_path` | A torchvision ResNet18 state dictionary, including the original classifier; features are extracted after loading the full backbone. |
| `latent.decoder_path`, `decoder_type: sdvae` | A local Diffusers `AutoencoderKL` directory, including configuration and weights. The multiplier must match the stored data convention. |
| `latent.decoder_path` and `latent.stat_path`, `decoder_type: vavae` | A converted Diffusers AutoencoderKL decoder and a PyTorch statistics file with `mean` and `std` tensors; see [the VA-VAE implementation](networks/vaes/vavae.py). The verified DiT recipes use this converted decoder with 32 channels and 16× downsampling; match the exact initialization, latent preprocessing and statistics. |

Explicit representation paths override the fallback paths under
`$DM_MODEL_ZOO/pretrained/`. Setting `DM_MODEL_ZOO` alone does not populate them.
Keep ImageFolder data fixed while reusing its cached file index; remove the
index or choose a fresh `data.cache_dir` after changing nested image contents.

## SD-VAE latent convention

The local safetensors loader returns `stored_latent * latent.multiplier`. The
SD-VAE decoder divides generated latents by that same multiplier. The public
templates use `0.18215`, four channels and a downsampling ratio of eight.
Store raw posterior means when following this convention; supplying latents
already scaled by `0.18215` would apply the factor twice. The optional NumPy
loader expects NHWC arrays in `<split>_moments.npy`,
`<split>_moments_flip.npy`, and `<split>_targets.npy`; despite the filenames,
it reads latent values directly and does not sample a posterior.

[UCGM's SD-VAE encoder](https://github.com/LINs-lab/UCGM/blob/main/autoencs/sdvae.py)
exports unscaled posterior means. It maps `sdvae_f8c4` to the MSE VAE and
`sdvae_ema_f8c4` to the EMA VAE. Its
[SD-VAE training configuration](https://github.com/LINs-lab/UCGM/blob/main/configs/training_multi_steps/in1k256_tit_xl_sdvae.yaml)
uses data from `sdvae_f8c4`, disables latent normalization, applies `0.18215`,
and selects an EMA decoder. The public templates retain that combination;
the encoder and decoder variant names are not interchangeable asset aliases.
When reproducing another checkpoint, follow its actual preprocessing and VAE
choices. Shape compatibility alone does not verify the latent distribution.

The upstream [data exporter](https://github.com/LINs-lab/UCGM/blob/main/data.py)
writes the three safetensors keys above. Its
[256-pixel launcher](https://github.com/LINs-lab/UCGM/blob/main/scripts/data/in1k256.sh)
iterates over several VAE configurations; select the intended SD-VAE variant
when preparing these templates instead of treating all outputs as equivalent.

## Evaluation assets

Set `DM_MODEL_ZOO` and `DM_DATA_ZOO` before using `scripts/eval.sh`.
`evaluation.models` chooses evaluation encoders independently of the training
representations. The public templates request Inception only.

Inception weights are cached at
`$DM_MODEL_ZOO/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth`,
or the path selected by `DM_INCEPTION_CKPT`. The loader can download missing
weights from the torch-fidelity release. Prepopulate this file for offline runs.
Reference statistics live in `$DM_DATA_ZOO/fid_stats`, or `DM_FID_STATS_DIR`:

- `jit_in256_stats.npz` produces the `FID(JiT)` row.
- `guided_diffusion_stats.npz` produces the `FID(ADM)` row.

At least one file must exist for Inception evaluation; an absent second file
is skipped with a warning. Each must contain `mu` and `sigma` for the matching
feature extractor and dataset. Other evaluation representations require their
own statistics and cached weights. Precision/recall additionally needs the
reference data selected by `evaluation.prc_ref_dir`.

## Validation still required

Local native-component checks use synthetic images and random frozen weights.
They establish component execution and controlled checkpoint restoration, not
the correctness of downloaded pretrained assets or ImageNet quality. Before
publishing a result, record source revisions, asset hashes, preprocessing,
class mapping and the exact configuration; validate the actual downloaded
weights, decoder, statistics and data subset together. AIStudio verification
is tracked separately from CPU checks and complete training results.
