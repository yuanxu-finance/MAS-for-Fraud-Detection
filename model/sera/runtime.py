"""Repository paths and prepared CSV loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL.parent
CONFIG = MODEL / "config"
DATASET = Path(os.environ.get("MAS_DATASET_DIR", str(ROOT / "dataset"))).resolve()
OUTPUT = Path(os.environ.get("MAS_OUTPUT_DIR", str(ROOT / "outputs"))).resolve()
CSV_SCHEMA = json.loads((CONFIG / "csv_schema.json").read_text(encoding="utf-8"))


def read_dataset_csv(path):
    """Restore prepared CSV column types from the supplied schema."""
    path = Path(path).with_suffix(".csv")
    schema = CSV_SCHEMA[path.parent.name]
    dtype = {
        key: (
            "string"
            if spec["dtype"].startswith("datetime") or spec["dtype"] == "category"
            else spec["dtype"]
        )
        for key, spec in schema.items()
    }
    frame = pd.read_csv(
        path,
        dtype=dtype,
        keep_default_na=False,
        na_values=["__MAS_NULL__"],
        float_precision="round_trip",
    )
    for column, spec in schema.items():
        if spec["dtype"].startswith("datetime"):
            frame[column] = pd.to_datetime(frame[column]).astype(spec["dtype"])
        elif spec["dtype"] == "category":
            frame[column] = pd.Categorical(
                frame[column], categories=spec["categories"], ordered=spec["ordered"]
            )
    return frame
