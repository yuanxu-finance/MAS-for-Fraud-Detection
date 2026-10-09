"""Prepare datasets and execute the SERA model pipeline."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict

from model import SERA
from . import datasets
from .config import DATASETS, get_path, load_model_config as load_config
from .datasets import Workspace, load_dataset, make_splits
from .features import fit_base_features
from .runtime import DATASET, OUTPUT


def build_workspace(cfg: Dict[str, Any], verbose: bool = True) -> Workspace:
    start = time.perf_counter()
    ds = load_dataset(cfg["dataset"], int(cfg.get("window_rows", 400000)))
    frame = ds.frame
    splits = make_splits(
        frame,
        get_path(cfg, "split.mode"),
        list(get_path(cfg, "split.ratios")),
        get_path(cfg, "split.group_column"),
        int(cfg.get("seed", 2026)),
    )
    y = frame["label"].to_numpy(dtype=int)
    splits.assert_label_coverage(y)
    space = fit_base_features(frame, splits.train)
    X_base = space.transform(frame)
    if verbose:
        print(
            f"  [Data] {ds.name}: {len(frame):,} rows, positive rate {ds.meta['positive_rate']:.6f}, span {ds.meta['time_span_days']:.1f} days, splits {splits.sizes()}"
        )
    return Workspace(
        cfg, ds, frame, splits, X_base, space.columns, y, time.perf_counter() - start
    )


def run(
    dataset="banksim", seed=2026, datasets_dir=None, output_dir=None, window_rows=None
):
    """Load a dataset, run SERA, and return its evaluation and diagnostics."""
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
    result = SERA(config).fit_evaluate(workspace, verbose=True)
    if result["contracts"]["n_failed"]:
        raise RuntimeError(f"SERA contract checks failed for {dataset}")
    result["model"] = "SERA"
    result["configuration"] = config
    return result
