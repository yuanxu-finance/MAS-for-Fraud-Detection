from __future__ import annotations
import copy
import json
from typing import Any, Dict, Mapping
from .runtime import CONFIG

_CONFIGS = json.loads((CONFIG / "presets.json").read_text(encoding="utf-8"))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge mappings, replacing lists and leaf values."""
    out = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_path(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(
    cfg: Mapping[str, Any], overrides: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply dotted-path overrides and reject unknown configuration paths."""
    out = copy.deepcopy(dict(cfg))
    for dotted, value in overrides.items():
        if get_path(out, dotted, _MISSING) is _MISSING:
            raise KeyError(f"Unknown configuration override path: {dotted}")
        set_path(out, dotted, value)
    return out


class _Missing:
    pass


_MISSING = _Missing()


def load_config(dataset, config_dir=None):
    if dataset not in _CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    cfg = deep_merge(_CONFIGS["default"], _CONFIGS[dataset])
    cfg["dataset"] = dataset
    validate(cfg)
    return cfg


def validate(cfg: Mapping[str, Any]) -> None:
    """Validate configuration ranges and required fields before execution."""
    alpha = float(get_path(cfg, "constraint.alpha"))
    delta = float(get_path(cfg, "constraint.delta"))
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("constraint.alpha and constraint.delta must be in (0,1)")
    if get_path(cfg, "constraint.method") not in ("np", "empirical", "fixed"):
        raise ValueError("constraint.method must be np, empirical, or fixed")
    eps = float(get_path(cfg, "graph.epsilon"))
    if not 0.0 < eps < 1.0:
        raise ValueError("graph.epsilon must be in (0,1)")
    if get_path(cfg, "graph.counter") not in ("exact", "cms"):
        raise ValueError("graph.counter must be exact or cms")
    if float(get_path(cfg, "graph.bucket_seconds")) <= 0:
        raise ValueError("graph.bucket_seconds must be positive")
    if get_path(cfg, "context.mode") not in ("soft", "hard"):
        raise ValueError("context.mode must be soft or hard")
    if int(get_path(cfg, "context.top_k")) <= 0:
        raise ValueError("context.top_k must be positive")
    if float(get_path(cfg, "context.window_seconds")) <= 0:
        raise ValueError("context.window_seconds must be positive")
    if str(get_path(cfg, "semantic.enabled")).lower() not in (
        "auto",
        "text_only",
        "true",
        "false",
    ):
        raise ValueError("semantic.enabled must be auto, text_only, true, or false")
    if get_path(cfg, "evaluation.selection_criterion") not in ("ap", "tpr_at_alpha"):
        raise ValueError("evaluation.selection_criterion must be ap or tpr_at_alpha")
    quantile = get_path(cfg, "graph.rho_quantile")
    if isinstance(quantile, str):
        if quantile.lower() != "auto":
            raise ValueError("graph.rho_quantile must be auto or a number in [0,1]")
        grid = get_path(cfg, "graph.rho_quantile_grid") or []
        if not grid or any((not 0.0 <= float(q) <= 1.0 for q in grid)):
            raise ValueError(
                "Automatic graph.rho_quantile requires a nonempty rho_quantile_grid with values in [0,1]"
            )
    elif not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("graph.rho_quantile must be in [0,1]")
    zeta = get_path(cfg, "graph.zeta_mode", "product")
    if str(zeta).lower() == "auto":
        grid = get_path(cfg, "graph.zeta_mode_grid") or []
        if not grid or any((g not in ("product", "burst_only") for g in grid)):
            raise ValueError("zeta_mode_grid values must be product or burst_only")
    elif zeta not in ("product", "burst_only"):
        raise ValueError("graph.zeta_mode must be auto, product, or burst_only")
    ratios = get_path(cfg, "split.ratios")
    if (
        len(ratios) != 4
        or any((r <= 0 for r in ratios))
        or abs(sum(ratios) - 1.0) > 1e-08
    ):
        raise ValueError(
            "split.ratios must contain four positive values summing to 1 (train/tune/calibration/test)"
        )
    if get_path(cfg, "split.mode") not in ("temporal", "group"):
        raise ValueError("split.mode must be temporal or group")
    if get_path(cfg, "split.mode") == "group" and (
        not get_path(cfg, "split.group_column")
    ):
        raise ValueError("split.group_column is required for group splits")
    if not get_path(cfg, "semantic.fields"):
        raise ValueError("semantic.fields must contain explicit field specifications")


def config_digest(cfg: Mapping[str, Any]) -> str:
    """Hash the configuration for reproducible result tracking."""
    import hashlib

    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode(
        "utf-8"
    )
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


_MODEL_CONFIGS = json.loads((CONFIG / "datasets.json").read_text(encoding="utf-8"))
DATASETS = tuple(_MODEL_CONFIGS)


def load_model_config(dataset: str, seed: int = 2026) -> Dict[str, Any]:
    """Return an independent copy of a full-model dataset configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}; choose from {DATASETS}")
    config = copy.deepcopy(_MODEL_CONFIGS[dataset])
    config["seed"] = seed
    return config
