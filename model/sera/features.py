from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

CATEGORICAL = ("cat1", "cat2", "cat3")


@dataclass
class BaseFeatureSpace:
    columns: List[str]
    category_maps: Dict[str, Dict[str, int]]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return _build_matrix(frame, self.columns, self.category_maps)


def _time_parts(ts: np.ndarray) -> Dict[str, np.ndarray]:
    seconds = np.asarray(ts, dtype=np.float64)
    day_seconds = np.mod(seconds, 86400.0)
    return {
        "hour_of_day": day_seconds / 3600.0,
        "day_of_week": np.mod(np.floor(seconds / 86400.0) + 4.0, 7.0),
        "is_night": ((day_seconds < 6 * 3600.0) | (day_seconds >= 22 * 3600.0)).astype(
            np.float64
        ),
    }


def _build_matrix(
    frame: pd.DataFrame, columns: List[str], category_maps: Dict[str, Dict[str, int]]
) -> np.ndarray:
    amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy(dtype=np.float64)
    parts: Dict[str, np.ndarray] = {
        "amount": amount,
        "log_amount": np.log1p(np.abs(amount)) * np.sign(amount),
        "amount_is_missing": np.isnan(amount).astype(np.float64),
    }
    parts.update(_time_parts(frame["ts"].to_numpy()))
    for col in CATEGORICAL:
        mapping = category_maps[col]
        values = frame[col].astype(str).to_numpy()
        parts[f"{col}_code"] = np.array(
            [mapping.get(v, -1) for v in values], dtype=np.float64
        )
    for col in frame.columns:
        if col.startswith("f_"):
            parts[col] = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                dtype=np.float64
            )
    matrix = np.column_stack([parts[name] for name in columns])
    return np.nan_to_num(matrix, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(
        np.float32
    )


def fit_base_features(frame: pd.DataFrame, train_idx: np.ndarray) -> BaseFeatureSpace:
    """Fit categorical encodings on the training segment only."""
    category_maps: Dict[str, Dict[str, int]] = {}
    for col in CATEGORICAL:
        values = sorted(set(frame[col].astype(str).to_numpy()[train_idx].tolist()))
        category_maps[col] = {v: i for i, v in enumerate(values)}
    columns = [
        "amount",
        "log_amount",
        "amount_is_missing",
        "hour_of_day",
        "day_of_week",
        "is_night",
    ]
    columns += [f"{c}_code" for c in CATEGORICAL]
    columns += [c for c in frame.columns if c.startswith("f_")]
    return BaseFeatureSpace(columns=columns, category_maps=category_maps)
