import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from networks.generator import UniversalAdapter
from networks import backbone as backbones
from networks.representors import (
    MAERepresentor,
    PixelRepresentor,
    ResNetRepresentor,
    SigLIP2Representor,
)
from networks.vaes import SDVAE, VAVAE


def build_decoder(latent_cfg, device):
    """Construct the optional frozen image decoder shared by all algorithms."""
    if latent_cfg is None:
        return None
    decoder_type = latent_cfg.get("decoder_type", "sdvae")
    if decoder_type == "sdvae":
        decoder = SDVAE(
            model_path=latent_cfg["decoder_path"],
            multiplier=latent_cfg.get("multiplier", 0.18215),
        )
    elif decoder_type == "vavae":
        decoder = VAVAE(
            model_path=latent_cfg["decoder_path"],
            stat_path=latent_cfg["stat_path"],
            multiplier=latent_cfg.get("multiplier", 1.0),
        )
    else:
        raise ValueError(f"Unknown latent.decoder_type={decoder_type!r}")
    return decoder.to(device)


def build_representation(cfg):
    """Construct a frozen encoder from the shared representation schema."""
    params = dict(cfg)
    arch = params.pop("arch")
    constructors = {
        "mae_b": MAERepresentor,
        "siglip2_b": SigLIP2Representor,
        "resnet18": ResNetRepresentor,
        "pixel": PixelRepresentor,
    }
    if arch not in constructors:
        raise ValueError(f"Unknown representation.arch={arch!r}")
    return constructors[arch](**params)


def validate_checkpoint_path(cfg):
    """Reject a missing selected checkpoint instead of starting fresh silently."""
    experiment = cfg["experiment"]
    key = "resume" if experiment.get("resume") else "init_weights"
    path = experiment.get(key)
    if path and not os.path.isfile(path):
        raise FileNotFoundError(f"experiment.{key} checkpoint does not exist: {path}")


def _check_keys(value, allowed, location):
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {location} keys: {', '.join(sorted(unknown))}")


def public_model_params(cfg):
    """Validate the public model section without rewriting the configuration."""
    model = cfg["model"]
    _check_keys(model, ("name", "params"), "model")
    params = model.get("params", {})
    _check_keys(
        params,
        (
            "lambda",
            "rho",
            "use_potential_grad",
            "representations",
            "critic",
            "feature_stats_decay",
            "trainer",
        ),
        "model.params",
    )
    if model["name"] != "tbsm" and any(
        k in params for k in ("lambda", "rho", "use_potential_grad")
    ):
        raise ValueError("lambda, rho and use_potential_grad apply only to TBSM")
    if not isinstance(params.get("use_potential_grad", False), bool):
        raise TypeError("use_potential_grad must be a bool")
    _check_keys(
        params.get("trainer", {}),
        (
            "gen_lr",
            "critic_lr",
            "gen_betas",
            "critic_betas",
            "critic_steps",
            "grad_accum_steps",
            "ema_decay",
            "t_sampling",
            "noise_scale",
        ),
        "model.params.trainer",
    )
    return params


def _critic_options(shared, override, name):
    allowed = (
        "arch",
        "hidden_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "time_dim",
        "use_spectral_norm",
        "use_cosine_attention",
    )
    _check_keys(shared, allowed, "model.params.critic")
    _check_keys(override, allowed, "representation.critic")
    options = {**shared, **override}
    arch = options.pop("arch", "scattering")
    width, heads = options.get("hidden_dim", 736), options.get("num_heads", 8)
    if (
        not isinstance(width, int)
        or not isinstance(heads, int)
        or min(width, heads) < 1
        or width % heads
    ):
        raise ValueError(
            "critic.hidden_dim must be positive and divisible by critic.num_heads"
        )
    switches = ("use_spectral_norm", "use_cosine_attention")
    for key in switches:
        if key in options and not isinstance(options[key], bool):
            raise TypeError(f"critic.{key} must be a bool")
    if name == "tbsm":
        if arch != "scattering" or any(k in options for k in switches):
            raise ValueError(
                "TBSM's native critic supports arch=scattering and capacity parameters only"
            )
    elif arch not in ("scattering", "naive", "scattering_variant"):
        raise ValueError(f"Unknown critic.arch={arch!r}")
    elif arch == "scattering" and any(k in options for k in switches):
        raise ValueError(
            "Attention/spectral-norm switches require critic.arch=scattering_variant"
        )
    elif arch == "naive":
        if set(options) - {"use_spectral_norm"}:
            raise ValueError("Naive critic supports only use_spectral_norm")
    return arch, options


def representation_options(params, name):
    """Return encoder arguments and head options with per-encoder overrides."""
    representations = params.get("representations", [])
    if not isinstance(representations, list):
        raise TypeError("model.params.representations must be a list")
    if not representations:
        raise ValueError("Training requires at least one representation")
    allowed_arches = ("mae_b", "siglip2_b", "resnet18")
    if name != "tbsm":
        allowed_arches += ("pixel",)
    entries = []
    for rep in representations:
        _check_keys(
            rep,
            (
                "arch",
                "input_size",
                "ckpt_path",
                "patch_size",
                "loss_weight",
                "feature_stats_decay",
                "critic",
            ),
            "representation",
        )
        if "arch" not in rep:
            raise ValueError("Each representation requires arch")
        if rep["arch"] not in allowed_arches:
            raise ValueError(
                f"Unknown representation.arch={rep['arch']!r} for {name}; expected {allowed_arches}"
            )
        if "patch_size" in rep and rep["arch"] != "pixel":
            raise ValueError("representation.patch_size applies only to arch=pixel")
        encoder = {
            k: rep[k]
            for k in ("arch", "input_size", "ckpt_path", "patch_size")
            if k in rep
        }
        arch, critic = _critic_options(
            params.get("critic", {}), rep.get("critic", {}), name
        )
        options = {
            "loss_weight": rep.get("loss_weight", 1.0),
            "ema_beta": rep.get(
                "feature_stats_decay", params.get("feature_stats_decay", 0.999)
            ),
            "critic": critic,
        }
        if name == "tbsm":
            options.update(
                intra_weight=params.get("lambda", 0.9),
                track_weight=params.get("rho", 0.9),
            )
            critic["use_potential_grad"] = params.get("use_potential_grad", False)
        else:
            options["critic_arch"] = arch
            critic["output_mode"] = "potential"
        entries.append((encoder, options))
    return entries


def training_options(params):
    """Read shared trainer settings as constructor arguments, not a legacy YAML."""
    trainer = params.get("trainer", {})
    for key in ("critic_steps", "grad_accum_steps"):
        value = trainer.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"model.params.trainer.{key} must be a positive integer")
    return {
        "gen_lr": trainer["gen_lr"],
        "critic_lr": trainer["critic_lr"],
        "ema_decay": trainer.get("ema_decay", 0.999),
        "t_sampling": trainer.get("t_sampling", [1.0]),
        "noise_scale": trainer.get("noise_scale", 1.0),
        "gen_betas": trainer.get("gen_betas", (0.9, 0.999)),
        "critic_betas": trainer.get("critic_betas", (0.9, 0.999)),
        "grad_accum_steps": trainer.get("grad_accum_steps", 1),
        "critic_steps": trainer.get("critic_steps", 1),
    }


def build_network(cfg):
    network_cfg = cfg["network"]
    _check_keys(network_cfg, ("name", "params", "adapter"), "network")
    params = network_cfg.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("network.params must be a mapping")
    adapter_cfg = network_cfg.get("adapter")
    if "adapter" in params:
        raise ValueError("network.params.adapter is removed; use network.adapter")
    if adapter_cfg is not None:
        _check_keys(
            adapter_cfg,
            (
                "source_prediction_type",
                "target_prediction_type",
            ),
            "network.adapter",
        )
    network_name = network_cfg["name"]
    if network_name not in backbones.__all__:
        raise ValueError(f"Unknown network: {network_name}")
    num_classes = cfg["data"]["num_classes"]
    constructor = getattr(backbones, network_name)
    backbone = constructor(num_classes=num_classes, **params)
    return UniversalAdapter(backbone, **adapter_cfg) if adapter_cfg else backbone


def input_shape_from_config(cfg):
    channels = 3
    resolution = cfg["data"]["resolution"]
    latent_cfg = cfg.get("latent")
    if latent_cfg is not None:
        channels = int(latent_cfg.get("channels", 4))
        resolution = int(resolution) // int(latent_cfg.get("downsample_ratio", 8))
    return channels, resolution, resolution


def wrap_ddp_if_needed(method_model, cfg):
    runtime_cfg = cfg["runtime"]
    if runtime_cfg["world_size"] <= 1:
        return method_model

    local_rank = runtime_cfg["local_rank"]
    method_model.generator.backbone = DDP(
        method_model.generator.backbone,
        device_ids=[local_rank],
    )
    for matcher in method_model.matchers:
        if matcher.critic is not None:
            matcher.critic = DDP(matcher.critic, device_ids=[local_rank])
    return method_model


def raw_module(module):
    return module.module if isinstance(module, DDP) else module


def extract_backbone_state(state):
    prefixes = [
        "generator.backbone.",
        "_eval_generator.backbone.",
        "backbone.",
    ]
    extracted = {}
    for key, value in state.items():
        name = key
        for prefix in prefixes:
            if key.startswith(prefix):
                name = key[len(prefix) :]
                break
        extracted[name] = value
    return extracted


def load_backbone_state(backbone, state):
    """Select a compatible prefix mapping before loading any model weights."""
    raw_backbone = raw_module(backbone)
    # Adapters may nest the network under .model or .backbone.model; retain
    # those initialization layouts without mutating the model during selection.
    def candidates():
        yield state
        for prefix in ("model.", "backbone.", "backbone.model."):
            yield {
                key if key.startswith(prefix) else f"{prefix}{key}": value
                for key, value in state.items()
            }
        yield {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in state.items()
        }

    target = raw_backbone.state_dict()
    target_keys = set(target)
    best_candidate = None
    best_score = None
    for candidate in candidates():
        candidate_keys = set(candidate)
        common = target_keys & candidate_keys
        if not common:
            continue
        if any(
            isinstance(target[key], torch.Tensor)
            and (
                not isinstance(candidate[key], torch.Tensor)
                or candidate[key].shape != target[key].shape
            )
            for key in common
        ):
            continue
        score = len(target_keys - candidate_keys) + len(candidate_keys - target_keys)
        if best_score is None or score < best_score:
            best_candidate = candidate
            best_score = score
        if score == 0:
            break
    if best_candidate is None:
        raise RuntimeError(
            "No compatible checkpoint weights for the configured backbone"
        )
    return raw_backbone.load_state_dict(best_candidate, strict=False)


def select_init_state(ckpt):
    for key in ("backbone", "model", "state_dict"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def training_checkpoint(model, step):
    """Capture optimizer state and pending gradients at any outer training step."""
    checkpoint = {
        "backbone": raw_module(model.generator.backbone).state_dict(),
        "model": model.state_dict(),
        "generator_optimizer": model.generator_optimizer.state_dict(),
        "step": step,
        "training_state": {
            "version": model.checkpoint_version,
            "grad_accum_steps": model.grad_accum_steps,
            "critic_steps": model.critic_steps,
            "gradients": {
                name.replace(".module.", "."): param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            },
        },
    }
    if model.critic_optimizer is not None:
        checkpoint["critic_optimizer"] = model.critic_optimizer.state_dict()
    return checkpoint


def _checkpoint_gradients(model, checkpoint):
    """Validate resume metadata before changing model weights or gradients."""
    step = checkpoint["step"]
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    for key in ("generator_optimizer", "critic_optimizer"):
        if getattr(model, key) is not None and key not in checkpoint:
            raise ValueError(
                f"Training resume requires {key}; use init_weights for initialization"
            )
    state = checkpoint.get("training_state")
    if (
        not isinstance(state, dict)
        or type(state.get("version")) is not int
        or state["version"] != model.checkpoint_version
    ):
        raise ValueError("Unsupported checkpoint training_state")
    for key in ("grad_accum_steps", "critic_steps"):
        if type(state.get(key)) is not int or state[key] != getattr(model, key):
            raise ValueError(
                f"Checkpoint {key} differs from the training configuration"
            )
    gradients = state.get("gradients")
    if not isinstance(gradients, dict):
        raise ValueError("Checkpoint gradients must be a mapping")
    if model.checkpoint_version == 2 and gradients:
        raise ValueError("GAN checkpoints require a completed generator update")
    parameters = dict(model.named_parameters())
    for name, gradient in gradients.items():
        param = parameters.get(name)
        if (
            param is None
            or not param.requires_grad
            or not isinstance(gradient, torch.Tensor)
            or gradient.shape != param.shape
            or gradient.dtype != param.dtype
        ):
            raise ValueError(f"Incompatible checkpoint gradient: {name}")
    return gradients


def load_checkpoint(method_model, cfg):
    exp_cfg = cfg["experiment"]
    device = cfg["runtime"]["device"]

    if exp_cfg["resume"]:
        ckpt = torch.load(exp_cfg["resume"], map_location=device, weights_only=True)
        gradients = _checkpoint_gradients(method_model, ckpt)
        # Checkpoints are saved after DDP wrapping, while resume loading happens
        # before wrapping. Remove DDP's module component so every tensor is
        # loaded instead of being silently skipped by strict=False.
        model_state = {
            key.replace(".module.", "."): value for key, value in ckpt["model"].items()
        }
        method_model.load_state_dict(model_state, strict=True)
        method_model.generator_optimizer.load_state_dict(ckpt["generator_optimizer"])
        if method_model.critic_optimizer is not None:
            method_model.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        for name, param in method_model.named_parameters():
            gradient = gradients.get(name)
            param.grad = None if gradient is None else gradient.to(param.device).clone()
        completed_steps = ckpt["step"]
        method_model.set_training_step(completed_steps)
        return completed_steps

    if exp_cfg["init_weights"]:
        ckpt = torch.load(
            exp_cfg["init_weights"], map_location=device, weights_only=True
        )
        state = select_init_state(ckpt)
        backbone_state = extract_backbone_state(state)
        result = load_backbone_state(method_model.generator.backbone, backbone_state)
        if result.missing_keys or result.unexpected_keys:
            print(
                "Loaded init_weights with "
                f"{len(result.missing_keys)} missing keys and "
                f"{len(result.unexpected_keys)} unexpected keys"
            )
        method_model.reset_ema()

    return 0


def finish_build(method_model, cfg):
    start_step = load_checkpoint(method_model, cfg)
    method_model = wrap_ddp_if_needed(method_model, cfg)
    return method_model, start_step
