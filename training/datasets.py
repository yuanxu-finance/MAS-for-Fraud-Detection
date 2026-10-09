from __future__ import annotations
import json
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .runtime import ROOT, MODEL, DATASET, OUTPUT, read_dataset_csv

CODE_DIR = str(MODEL)
RUNNING_DIR = str(ROOT)
DATA_DIR = str(DATASET)
CACHE_DIR = str(OUTPUT / "cache")
CANON_COLUMNS = (
    "ts",
    "entity",
    "counterparty",
    "amount",
    "cat1",
    "cat2",
    "cat3",
    "text",
    "label",
)


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    meta: Dict[str, Any]

    @property
    def numeric_extra(self) -> List[str]:
        return [c for c in self.frame.columns if c.startswith("f_")]

    def __repr__(self) -> str:
        return f"<Dataset {self.name} rows={len(self.frame)} pos={self.frame.label.mean():.5f}>"


def _unix_seconds(s: pd.Series) -> pd.Series:
    """Convert datetimes to Unix seconds regardless of their stored time unit."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[s]").astype("int64")
    return pd.to_numeric(s, errors="coerce").astype("int64")


def _middle_window(d: pd.DataFrame, ts_col: str, n: int) -> pd.DataFrame:
    d = (
        d.dropna(subset=[ts_col])
        .sort_values(ts_col, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(d) <= n:
        return d
    lo = (len(d) - n) // 2
    return d.iloc[lo : lo + n].reset_index(drop=True)


def _finalize(name: str, d: pd.DataFrame, meta: Dict[str, Any]) -> Dataset:
    d = d.reset_index(drop=True)
    for col in CANON_COLUMNS:
        if col not in d.columns:
            d[col] = "" if col == "text" else np.nan
    for col in ("cat1", "cat2", "cat3"):
        d[col] = d[col].astype(str)
    d["text"] = d["text"].fillna("").astype(str)
    d["label"] = d["label"].astype(int)
    d = d.sort_values("ts", kind="mergesort").reset_index(drop=True)
    meta = dict(meta)
    meta.update(
        rows=int(len(d)),
        positive_rate=float(d.label.mean()),
        n_positive=int(d.label.sum()),
        entities=int(d.entity.nunique()),
        counterparties=int(d.counterparty.nunique()),
        has_text=bool((d.text.str.len() > 0).any()),
        time_span_days=float((d.ts.max() - d.ts.min()) / 86400.0),
    )
    return Dataset(name, d, meta)


def _intent_f(v: Any, default: float = np.nan) -> float:
    try:
        return default if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return default


def _intent_json(fn: str) -> Any:
    with open(os.path.join(DATA_DIR, "intent-tx-18k", fn), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_ibm_aml(window: int) -> Dataset:
    """Load IBM AML HI-Small from the archive or uncompressed CSV."""
    folder = os.path.join(DATA_DIR, "ibm-aml")
    archive = os.path.join(folder, "transactions_full.csv.zip")
    plain = os.path.join(folder, "HI-Small_Trans.csv")
    if os.path.exists(archive):
        with zipfile.ZipFile(archive) as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(inner) as fh:
                raw = pd.read_csv(fh)
    elif os.path.exists(plain):
        raw = pd.read_csv(plain)
    else:
        raise FileNotFoundError(
            f"IBM AML HI-Small transaction data not found at either path: {archive} and {plain}. Provide the archive or uncompressed CSV."
        )
    full_rows, full_rate = (len(raw), float(raw["Is Laundering"].mean()))
    raw["_ts"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw = _middle_window(raw, "_ts", window)
    d = pd.DataFrame(
        dict(
            ts=_unix_seconds(raw["_ts"]),
            entity=raw["Account"].astype(str),
            counterparty=raw["Account.1"].astype(str),
            amount=pd.to_numeric(raw["Amount Paid"], errors="coerce"),
            cat1=raw["Payment Format"].astype(str),
            cat2="BANK" + raw["To Bank"].astype(str),
            cat3=raw["Receiving Currency"].astype(str),
            text="",
            label=raw["Is Laundering"].astype(int),
        )
    )
    d["f_amount_received"] = pd.to_numeric(
        raw["Amount Received"], errors="coerce"
    ).values
    d["f_cross_currency"] = (
        (raw["Receiving Currency"] != raw["Payment Currency"]).astype(int).values
    )
    d["f_cross_bank"] = (raw["From Bank"] != raw["To Bank"]).astype(int).values
    return _finalize(
        "ibm-aml",
        d,
        dict(
            source="github.com/IBM/AML-Data(CDLA-Sharing-1.0); Altman et al., NeurIPS 2023 D&B",
            task="Anti-money laundering (positive class: Is Laundering)",
            time_unit="Unix seconds",
            note=f"Full dataset: {full_rows:,} rows; positive rate: {full_rate:.6f}; uses a middle window; family labels are in pattern_labels.csv",
        ),
    )


def load_aml_pattern_labels() -> pd.DataFrame:
    """Join family labels by timestamp, accounts, and payment amount."""
    d = pd.read_csv(os.path.join(DATA_DIR, "ibm-aml", "pattern_labels.csv"))
    d["_t"] = pd.to_datetime(d.Timestamp, errors="coerce")
    d["ts"] = _unix_seconds(d["_t"])
    d["amt"] = pd.to_numeric(d.amt_paid, errors="coerce").round(2)
    if "family" not in d.columns:
        d["family"] = d.pattern.str.split(":").str[0].str.strip()
    d["family"] = (
        d["family"].str.replace("\\s*MAX\\s+\\d+.*$", "", regex=True).str.strip()
    )
    return d.rename(columns={"acct_from": "entity", "acct_to": "counterparty"})


def load_sparkov(window: int) -> Dataset:
    raw = read_dataset_csv(
        os.path.join(DATA_DIR, "sparkov", "transactions_full.parquet")
    )
    full_rows, full_rate = (len(raw), float(raw["is_fraud"].mean()))
    raw = _middle_window(raw, "unix_time", window)
    d = pd.DataFrame(
        dict(
            ts=raw["unix_time"].astype(float),
            entity=raw["cc_num"].astype(str),
            counterparty=raw["merchant"].astype(str),
            amount=raw["amt"].astype(float),
            cat1=raw["category"].astype(str),
            cat2=raw["state"].astype(str),
            cat3=raw["job"].astype(str),
            text="",
            label=raw["is_fraud"].astype(int),
        )
    )
    d["f_city_pop"] = pd.to_numeric(raw["city_pop"], errors="coerce").values
    return _finalize(
        "sparkov",
        d,
        dict(
            source="Sparkov generator; HF Nooha/cc_fraud_detection_dataset",
            task="Credit-card transaction fraud",
            time_unit="Unix seconds",
            note=f"Full dataset: {full_rows:,} rows; positive rate: {full_rate:.6f}; generated card transactions",
        ),
    )


def load_ieee_cis(window: int) -> Dataset:
    """Load IEEE-CIS using billing region as the counterparty proxy."""
    path = os.path.join(DATA_DIR, "ieee-cis", "ieee_cis_fraud_features.parquet")
    raw = read_dataset_csv(path)
    full_rows, full_rate = (len(raw), float(raw["is_fraud"].mean()))
    ts = raw["transaction_ts"]
    raw = raw.assign(_ts=_unix_seconds(ts).astype(float))
    raw = _middle_window(raw, "_ts", window)
    d = pd.DataFrame(
        dict(
            ts=raw["_ts"].astype(float),
            entity="C" + raw["card1"].astype(str),
            counterparty="A" + raw["addr1"].astype(str),
            amount=raw["transaction_amt"].astype(float),
            cat1=raw["product_cd"].astype(str),
            cat2=raw["card4"].astype(str),
            cat3=raw["card6"].astype(str),
            text="",
            label=raw["is_fraud"].astype(int),
        )
    )
    for col in ("dist1", "C1", "C13", "D1", "D15"):
        if col in raw.columns:
            d[f"f_{col.lower()}"] = pd.to_numeric(raw[col], errors="coerce").values
    return _finalize(
        "ieee-cis",
        d,
        dict(
            source="IEEE-CIS Fraud Detection(Kaggle 2019 / Vesta)",
            task="Credit-card transaction fraud",
            time_unit="Unix seconds derived from TransactionDT",
            note=f"Full dataset: {full_rows:,} rows; positive rate: {full_rate:.6f}; counterparty uses addr1 (billing region); graph nodes represent cards and regions",
        ),
    )


def _load_ibm_variant(
    name: str, folder: str, filename: str, note: str, window: int
) -> Dataset:
    """Load an IBM AML variant using the shared transaction schema."""
    path = os.path.join(DATA_DIR, folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} transaction file not found: {path}")
    raw = pd.read_csv(path)
    full_rows, full_rate = (len(raw), float(raw["Is Laundering"].mean()))
    raw["_ts"] = _unix_seconds(pd.to_datetime(raw["Timestamp"], errors="coerce"))
    raw = _middle_window(raw, "_ts", window)
    d = pd.DataFrame(
        dict(
            ts=raw["_ts"].astype(float),
            entity=raw["Account"].astype(str),
            counterparty=raw["Account.1"].astype(str),
            amount=pd.to_numeric(raw["Amount Paid"], errors="coerce"),
            cat1=raw["Payment Format"].astype(str),
            cat2="BANK" + raw["To Bank"].astype(str),
            cat3=raw["Receiving Currency"].astype(str),
            text="",
            label=raw["Is Laundering"].astype(int),
        )
    )
    return _finalize(
        name,
        d,
        dict(
            source="IBM AML (Altman et al., NeurIPS 2023 D&B); Kaggle ealtman2019",
            task="Anti-money laundering",
            time_unit="Unix seconds derived from Timestamp",
            note=f"Full dataset: {full_rows:,} rows; positive rate: {full_rate:.6f}. {note}",
        ),
    )


def load_ibm_aml_li(window: int) -> Dataset:
    """Load the IBM AML LI-Small variant."""
    return _load_ibm_variant(
        "ibm-aml-li",
        "ibm-aml-li",
        "LI-Small_Trans.csv",
        "LI-Small uses the same generator as HI-Small with lower illicit prevalence",
        window,
    )


def load_ibm_aml_medium(window: int) -> Dataset:
    """Load the IBM AML HI-Medium variant."""
    return _load_ibm_variant(
        "ibm-aml-medium",
        "ibm-aml-medium",
        "HI-Medium_Trans.csv",
        "HI-Medium variant",
        window,
    )


def load_banksim(window: int) -> Dataset:
    """Load BankSim and strip quotes from categorical fields."""
    path = os.path.join(DATA_DIR, "banksim", "bs140513_032310.csv")
    raw = pd.read_csv(path)
    strip = lambda s: s.astype(str).str.strip().str.strip("'")
    full_rows, full_rate = (len(raw), float(raw["fraud"].mean()))
    raw["_ts"] = pd.to_numeric(raw["step"], errors="coerce").astype(float) * 86400.0
    raw = _middle_window(raw, "_ts", window)
    d = pd.DataFrame(
        dict(
            ts=raw["_ts"].astype(float),
            entity=strip(raw["customer"]),
            counterparty=strip(raw["merchant"]),
            amount=pd.to_numeric(raw["amount"], errors="coerce"),
            cat1=strip(raw["category"]),
            cat2="AGE" + strip(raw["age"]),
            cat3="G" + strip(raw["gender"]),
            text="",
            label=raw["fraud"].astype(int),
        )
    )
    return _finalize(
        "banksim",
        d,
        dict(
            source="BankSim(Kaggle ealaxi/banksim1)",
            task="Retail card-payment fraud",
            time_unit="Unix seconds derived from daily steps",
            note=f"Full dataset: {full_rows:,} rows; positive rate: {full_rate:.6f}; quotes stripped from categorical fields",
        ),
    )


LOADERS = {
    "ibm-aml": load_ibm_aml,
    "sparkov": load_sparkov,
    "ieee-cis": load_ieee_cis,
    "ibm-aml-li": load_ibm_aml_li,
    "ibm-aml-medium": load_ibm_aml_medium,
    "banksim": load_banksim,
}


def load_dataset(
    name: str, window_rows: int = 400000, use_cache: bool = True
) -> Dataset:
    """Load a dataset with a cache keyed by its window size."""
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset {name!r}; available: {sorted(LOADERS)}")
    cache_frame = os.path.join(CACHE_DIR, f"{name}_w{window_rows}.parquet")
    cache_meta = os.path.join(CACHE_DIR, f"{name}_w{window_rows}.meta.json")
    if use_cache and os.path.exists(cache_frame) and os.path.exists(cache_meta):
        frame = pd.read_parquet(cache_frame)
        with open(cache_meta, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return Dataset(name, frame, meta)
    ds = LOADERS[name](window_rows)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        ds.frame.to_parquet(cache_frame, index=False)
        with open(cache_meta, "w", encoding="utf-8") as fh:
            json.dump(ds.meta, fh, ensure_ascii=False, indent=2)
    return ds


@dataclass(frozen=True)
class Splits:
    """Disjoint training, tuning, calibration, and test indices."""

    train: np.ndarray
    tune: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    mode: str
    boundaries: Tuple[float, ...] = ()

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "train": self.train,
            "tune": self.tune,
            "calibration": self.calibration,
            "test": self.test,
        }

    def sizes(self) -> Dict[str, int]:
        return {k: int(v.size) for k, v in self.as_dict().items()}

    def assert_valid(self, ts: np.ndarray) -> int:
        """Validate nonempty, disjoint splits and temporal ordering."""
        groups = self.as_dict()
        seen: set = set()
        total = 0
        for name, idx in groups.items():
            if idx.size == 0:
                raise ValueError(f"Split {name!r} is empty")
            overlap = seen & set(idx.tolist())
            if overlap:
                raise ValueError(
                    f"Samples overlap between splits: {sorted(overlap)[:5]}"
                )
            seen |= set(idx.tolist())
            total += int(idx.size)
        if self.mode == "temporal":
            order = ["train", "tune", "calibration", "test"]
            for left, right in zip(order, order[1:]):
                if float(ts[groups[left]].max()) >= float(ts[groups[right]].min()):
                    raise ValueError(
                        f"Temporal leakage: max({left}) must be strictly earlier than min({right})"
                    )
        return total

    def assert_label_coverage(self, y: np.ndarray) -> int:
        """Require positives in train, tune, and test, and negatives in calibration."""
        for name in ("train", "tune", "test"):
            idx = getattr(self, name)
            if int(y[idx].sum()) == 0:
                raise ValueError(
                    f"Split {name!r} has no positive samples (n={idx.size}); increase window_rows or select another window"
                )
        if int((y[self.calibration] == 0).sum()) == 0:
            raise ValueError(
                "Calibration requires negative samples for order-statistic thresholds"
            )
        return int(y.size)


def make_splits(
    frame: pd.DataFrame,
    mode: str,
    ratios: List[float],
    group_column: Optional[str] = None,
    seed: int = 2026,
) -> Splits:
    n = len(frame)
    cuts = np.cumsum(ratios)[:3]
    if mode == "temporal":
        ts = frame["ts"].to_numpy(dtype=np.float64)
        order = np.argsort(ts, kind="mergesort")
        ts_sorted = ts[order]
        unique_ts = np.unique(ts_sorted)
        if unique_ts.size < 4:
            raise ValueError(
                f"Only {unique_ts.size} distinct timestamps; four temporal segments are required"
            )
        cum = np.searchsorted(ts_sorted, unique_ts, side="right")
        picks: List[int] = []
        for target in cuts * n:
            candidate = int(np.argmin(np.abs(cum - target)))
            if picks:
                candidate = max(candidate, picks[-1] + 1)
            picks.append(min(candidate, unique_ts.size - 2))
        if not picks[0] < picks[1] < picks[2]:
            raise ValueError("Cannot find three distinct temporal split boundaries")
        b0, b1, b2 = (float(unique_ts[p]) for p in picks)
        parts = [
            np.flatnonzero(ts <= b0),
            np.flatnonzero((ts > b0) & (ts <= b1)),
            np.flatnonzero((ts > b1) & (ts <= b2)),
            np.flatnonzero(ts > b2),
        ]
        splits = Splits(
            parts[0], parts[1], parts[2], parts[3], mode, (b0, b1, b2, float(ts.max()))
        )
    elif mode == "group":
        if group_column is None or group_column not in frame.columns:
            raise ValueError(
                f"Group splits require an existing group_column; received {group_column!r}"
            )
        keys = frame[group_column].astype(str).to_numpy()
        uniq = np.array(sorted(set(keys.tolist())))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(uniq.size)
        positions = (cuts * uniq.size).astype(int)
        buckets = np.split(perm, positions)
        assign = {}
        for part_id, bucket in enumerate(buckets):
            for u in uniq[bucket]:
                assign[u] = part_id
        part_of = np.array([assign[k] for k in keys])
        splits = Splits(*[np.flatnonzero(part_of == i) for i in range(4)], mode=mode)
    else:
        raise ValueError(f"Unknown split mode {mode!r}")
    splits.assert_valid(frame["ts"].to_numpy())
    return splits


@dataclass
class Workspace:
    """Dataset, splits, and base features shared across configurations."""

    cfg: Dict[str, Any]
    dataset: Dataset
    frame: pd.DataFrame
    splits: Splits
    X_base: np.ndarray
    base_names: List[str]
    y: np.ndarray
    load_seconds: float = 0.0
    cache: Dict[Tuple[str, str], Any] = field(default_factory=dict)
