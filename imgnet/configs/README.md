# Configurations

Six general templates: `<model>_pixel_256.yaml` (JiT-B) and
`<model>_latent_256.yaml` (DiT-XL/2 + SD-VAE).

| `model.name` | Method |
| --- | --- |
| `tbsm` | Original vector supervision |
| `logistic_gan` | Non-saturating logistic GAN |
| `nextgan` | Potential-gradient field regression |

`reproduced/nextgan_dit_vavae_256.yaml` records the included 100K DiT-XL/1 +
VA-VAE experiment recipe. It uses placeholders for local asset paths and starts
fresh from the pretrained generator. This is separate from the six templates.

`network.name` selects the backbone. Replace paths and adjust GPU count and
batch size before running. Templates are starting configurations, not reproduced result recipes.
Representation paths expand `${DM_MODEL_ZOO}/pretrained/...`; set `DM_MODEL_ZOO`
to your local model cache, or replace `ckpt_path` with an explicit local path.

Defaults omitted from the YAML files:

| Setting | Default |
| --- | --- |
| Trainer `gen_betas`, `critic_betas` | [0.9, 0.999] |
| Trainer `critic_steps`, `grad_accum_steps` | 1 |
| Trainer `ema_decay`, `t_sampling`, `noise_scale` | 0.999, [1.0], 1.0 |
| Representation `loss_weight`, `feature_stats_decay` | 1.0, 0.999 |
| Critic `arch`, `hidden_dim`, `depth`, `num_heads` | scattering, 736, 4, 8 |
| Critic `mlp_ratio`, `time_dim` | 4.0, 128 |

The NextGAN pixel and latent templates explicitly set `critic_steps: 2` and
`grad_accum_steps: 1`.

The NextGAN JiT pixel template uses `label_drop_prob: 0.1` and
`critic_lr: 5e-5`. Latent templates also use `5e-5`; DiT defaults to class
dropout zero. Recorded DiT results and the included recipe are linked in
[the results table](../README.md#included-dit-recipe).

Trainer settings live under `model.params.trainer`; `gen_lr` and `critic_lr`
are required. Representation entries can override shared critic settings.
`arch` selects the representation once in the builder. Image encoders accept
`input_size` and `ckpt_path`; `pixel` uses only `patch_size` for its input features.
TBSM uses normalized bearings, `lambda: 0.9`, `rho: 0.9` and
`use_potential_grad: false` by default.

The shared time convention is fixed: `t=0` is data and `t=1` is noise.
`network.adapter.source_prediction_type` describes the backbone output;
the target prediction defaults to `x`. Prediction types use exactly `x`,
`noise` or `velocity`, without aliases. JiT handles its pretrained time
convention inside its own adapter. Interpolation direction is not configurable.

For LogisticGAN and NextGAN, one `training.total_steps` step completes one generator
update and `critic_steps` critic updates. Each critic update averages
`grad_accum_steps` fresh microbatches: one generator step consumes `K * A`
microbatches, with new data, noise, generator time and interpolation per batch.
Features are extracted and normalized once per real/fake batch.

LogisticGAN and NextGAN use the updated critic for the final generator loss.
Optimizers remain AdamW as configured. Spectral-normalization state is fixed
during generator queries. NextGAN samples an independent interpolation
coefficient per sample and representation branch in each critic update.

TBSM retains microbatch step counting: its critic updates every `A`
microbatches, and G updates once every `K` accumulation windows.

`experiment.init_weights` starts a new optimizer run; `experiment.resume`
restores the current checkpoint format (including `training_state`) and requires
the same accumulation/critic schedule. Older training checkpoint layouts are not supported.
GAN schedule checkpoints use `training_state.version: 2`, saved only after a
complete generator update; the previous GAN schedule cannot be resumed.
TBSM retains version 1 with pending gradients.
All methods use `critic_lr`, `critic_betas` and `critic_steps`; checkpoints store
the optimizer as `critic_optimizer` and network state under `matchers.*.critic.*`.
RNG state and data position are not restored. `evaluation.weights` selects `online` or `ema`.

Evaluation selects `evaluation.checkpoint`, then `experiment.resume`, then the
highest numbered `<model>_step_<number>.pt` in `experiment.save_dir`.
`evaluation.save_dir` defaults to `<experiment.save_dir>/eval/<checkpoint stem>`.
The main evaluation settings are:

| Setting | Default |
| --- | --- |
| `batch_size`, `num_images` | 256, 50000 |
| `dtype` | bf16 (`fp16` and `fp32` also supported) |
| `no_prc`, `eval_mmd` | false, false |
| `prc_models` | Same as `models` |
| `prc_k`, `prc_batch_size` | 3, 10000 |

The templates explicitly select `models: [inception]`. Without a selection,
evaluation uses Inception, MAE, DINOv2, CLIP, SigLIP and ConvNeXt.
Generated labels cycle through all `data.num_classes` classes across ranks.

The historical ImageNet comparison protocol uses EMA, 50000 images, seed 1,
eight GPU ranks and evaluation batch size 256 per GPU. The NextGAN latent
template uses this evaluation batch size. Keep batch size and GPU count
fixed when comparing checkpoints: CUDA noise is drawn per batch, so the same
seed with a different batch size does not produce the same sample set.
`FID(ADM)` and `FID(JiT)` use different reference statistics; compare the same
metric label. These evaluation settings do not replace the full paper recipe.

Evaluation does not require training representations, critics or optimizer settings.

[Usage](../README.md) · [Assets](../ASSETS.md)
