"""Public entry point for the SERA fraud-detection pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .sera import datasets
from .sera.pipeline import build_workspace, run_arm
from .sera.runtime import CONFIG, DATASET, OUTPUT

_CONFIGURATIONS = json.loads((CONFIG / "datasets.json").read_text(encoding="utf-8"))
DATASETS = tuple(_CONFIGURATIONS)


def load_config(dataset, seed=2026):
    """Return an independent copy of the dataset configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}; choose from {DATASETS}")
    config = copy.deepcopy(_CONFIGURATIONS[dataset])
    config["seed"] = seed
    return config


def run(
    dataset="banksim", seed=2026, datasets_dir=None, output_dir=None, window_rows=None
):
    """Fit SERA, calibrate on held-out negatives, and evaluate the test segment."""
    root = Path(datasets_dir or DATASET).resolve()
    output = Path(output_dir or OUTPUT).resolve()
    datasets.DATA_DIR = str(root)
    key = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    datasets.CACHE_DIR = str(output / "cache" / key)
    config = load_config(dataset, seed)
    if window_rows is not None:
        if window_rows <= 0:
            raise ValueError("window_rows must be positive")
        config["window_rows"] = window_rows
    workspace = build_workspace(config, verbose=True)
    result = run_arm(workspace, config, arm="full", verbose=True)
    if result["contracts"]["n_failed"]:
        raise RuntimeError(f"SERA contract checks failed for {dataset}")
    result["model"] = "SERA"
    result["configuration"] = config
    return result
