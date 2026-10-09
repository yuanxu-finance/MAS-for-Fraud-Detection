from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .contracts import SemanticSpec, TemporalAvailabilityGuard

_TOKEN_RE = re.compile("[\\w.:+-]+")


def _hash_token(token: str, dim: int) -> Tuple[int, float]:
    raw = int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
    )
    return (raw % dim, 1.0 if raw >> 17 & 1 == 0 else -1.0)


def _accumulate(vector: np.ndarray, tokens: Sequence[str]) -> None:
    dim = vector.shape[-1]
    for token in tokens:
        index, sign = _hash_token(token, dim)
        vector[index] += sign


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text).lower())


@dataclass
class SemanticArtifacts:
    """Semantic features and their separate audit metadata."""

    matrix: np.ndarray
    enabled: bool
    reason: str
    excluded_by_omega: Tuple[str, ...]
    serialized_columns: Tuple[str, ...]
    example_serialization: str = ""
    n_rows_checked: int = 0


class DomainSemanticEncoder:

    def __init__(self, gamma: SemanticSpec, cfg: Dict[str, Any]) -> None:
        self.gamma = gamma
        self.cfg = cfg
        self.dim = int(cfg.get("embedding_dim", 64))
        self.numeric_bins = int(cfg.get("numeric_bins", 32))
        self.max_text_tokens = int(cfg.get("max_text_tokens", 256))
        self.guard = TemporalAvailabilityGuard(gamma)
        self._numeric_edges: Dict[str, np.ndarray] = {}
        self._value_tables: Dict[str, Tuple[Dict[str, int], np.ndarray]] = {}
        self._constant: Optional[np.ndarray] = None
        self._columns: List[str] = []
        self._excluded: List[str] = []

    def _visible_columns(
        self, frame: pd.DataFrame, decision_offset: float = 0.0
    ) -> Tuple[List[str], List[str]]:
        """Select columns available at the transaction decision time."""
        candidates = [
            c for c in frame.columns if c in self.gamma.fields and c != "label"
        ]
        visible, excluded = ([], [])
        for name in candidates:
            spec = self.gamma.field(name)
            if not spec.in_serialization:
                continue
            if spec.posthoc or spec.availability_lag_seconds > decision_offset:
                excluded.append(name)
            else:
                visible.append(name)
        return (visible, excluded)

    def _static_tokens(self, name: str) -> List[str]:
        spec = self.gamma.field(name)
        if self.gamma.is_empty_spec():
            return [f"field:{name}"]
        tokens = [f"field:{name}"]
        tokens += [f"desc:{w}" for w in _tokenize(spec.description)]
        if spec.unit:
            tokens.append(f"unit:{spec.unit}")
        if spec.entity_role:
            tokens.append(f"role:{spec.entity_role}")
        if spec.source:
            tokens.append(f"source:{spec.source}")
        return tokens

    def _value_tokens(self, name: str, raw: Any, decoded: str) -> List[str]:
        if self.gamma.is_empty_spec():
            return [f"{name}={raw}"]
        tokens = [f"{name}={raw}"]
        if decoded != str(raw):
            tokens += [f"{name}:decoded:{w}" for w in _tokenize(decoded)]
        return tokens

    def fit(self, frame: pd.DataFrame, train_idx: np.ndarray) -> None:
        """Fit numeric bins and categorical vocabularies on training data."""
        visible, excluded = self._visible_columns(frame)
        self._columns, self._excluded = (visible, excluded)
        constant = np.zeros(self.dim, dtype=np.float64)
        for name in visible:
            _accumulate(constant, self._static_tokens(name))
        if not self.gamma.is_empty_spec():
            _accumulate(
                constant, [f"task:{w}" for w in _tokenize(self.gamma.task_description)]
            )
        self._constant = constant.astype(np.float32)
        train_frame = frame.iloc[train_idx]
        for name in visible:
            series = train_frame[name]
            if pd.api.types.is_numeric_dtype(frame[name]):
                values = pd.to_numeric(train_frame[name], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                values = values[np.isfinite(values)]
                if values.size == 0:
                    edges = np.array([0.0])
                else:
                    qs = np.linspace(0.0, 1.0, self.numeric_bins + 1)[1:-1]
                    edges = np.unique(np.quantile(values, qs))
                self._numeric_edges[name] = edges
                labels = [f"bin{i}" for i in range(edges.size + 1)] + ["binNA"]
                self._value_tables[name] = self._make_table(name, labels, decode=False)
            else:
                vocab = sorted(set(series.astype(str).tolist()))
                self._value_tables[name] = self._make_table(name, vocab, decode=True)

    def _make_table(
        self, name: str, values: Sequence[str], decode: bool
    ) -> Tuple[Dict[str, int], np.ndarray]:
        spec = self.gamma.field(name)
        table = np.zeros((len(values), self.dim), dtype=np.float32)
        index: Dict[str, int] = {}
        for row, value in enumerate(values):
            index[value] = row
            decoded = spec.decode(value) if decode else value
            _accumulate(table[row], self._value_tokens(name, value, decoded))
        return (index, table)

    def _codes(self, name: str, series: pd.Series) -> np.ndarray:
        index, _ = self._value_tables[name]
        if name in self._numeric_edges:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
            edges = self._numeric_edges[name]
            bins = np.searchsorted(edges, values, side="right")
            bins = np.where(np.isfinite(values), bins, edges.size + 1)
            labels = np.array([f"bin{i}" for i in range(edges.size + 1)] + ["binNA"])
            return np.array([index[label] for label in labels[bins]], dtype=np.int64)
        keys = series.astype(str).to_numpy()
        return np.array([index.get(k, -1) for k in keys], dtype=np.int64)

    def _extend_table(self, name: str, unseen: Sequence[str]) -> None:
        """Encode unseen categorical values in the existing hash space."""
        index, table = self._value_tables[name]
        spec = self.gamma.field(name)
        rows = np.zeros((len(unseen), self.dim), dtype=np.float32)
        for offset, value in enumerate(unseen):
            index[value] = table.shape[0] + offset
            _accumulate(
                rows[offset], self._value_tokens(name, value, spec.decode(value))
            )
        self._value_tables[name] = (index, np.vstack([table, rows]))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self._constant is None:
            raise RuntimeError("Call fit before transform")
        n = len(frame)
        out = np.repeat(self._constant[None, :], n, axis=0)
        for name in self._columns:
            if name not in self._numeric_edges:
                index, _ = self._value_tables[name]
                keys = frame[name].astype(str).to_numpy()
                unseen = sorted(set(keys.tolist()) - set(index))
                if unseen:
                    self._extend_table(name, unseen)
            codes = self._codes(name, frame[name])
            _, table = self._value_tables[name]
            known = codes >= 0
            if known.any():
                out[known] += table[codes[known]]
        if "text" in frame.columns:
            text = frame["text"].astype(str)
            for i in np.flatnonzero(text.str.len().to_numpy() > 0):
                tokens = _tokenize(text.iat[int(i)])[: self.max_text_tokens]
                _accumulate(out[int(i)], [f"text:{w}" for w in tokens])
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-08)).astype(np.float32)

    def serialize_row(self, frame: pd.DataFrame, row: int) -> str:
        """Serialize a row for audit output and reference checks."""
        record = frame.iloc[row]
        if self.gamma.is_empty_spec():
            pairs = [f"{name}={record[name]}" for name in self._columns]
            if str(record.get("text", "")):
                pairs.append(f"text={record['text']}")
            return " | ".join(pairs)
        units: List[str] = []
        for name in self._columns:
            spec = self.gamma.field(name)
            raw = record[name]
            units.append(
                f"[name={spec.name}; description={spec.description}; value={raw}; decoded={spec.decode(raw)}; unit={spec.unit}; role={spec.entity_role}; source={spec.source}; available_at={spec.available_at(float(record['ts']))}]"
            )
        if str(record.get("text", "")):
            units.append(f"[source=raw_text; text={record['text']}]")
        units.append(f"[task={self.gamma.task_description}]")
        return " ".join(units)

    def encode_row_reference(self, frame: pd.DataFrame, row: int) -> np.ndarray:
        """Compute reference row-wise token hashes for encoder validation."""
        record = frame.iloc[row]
        vector = np.zeros(self.dim, dtype=np.float64)
        for name in self._columns:
            _accumulate(vector, self._static_tokens(name))
        if not self.gamma.is_empty_spec():
            _accumulate(
                vector, [f"task:{w}" for w in _tokenize(self.gamma.task_description)]
            )
        for name in self._columns:
            spec = self.gamma.field(name)
            if name in self._numeric_edges:
                edges = self._numeric_edges[name]
                value = pd.to_numeric(
                    pd.Series([record[name]]), errors="coerce"
                ).to_numpy()[0]
                label = (
                    f"bin{int(np.searchsorted(edges, value, side='right'))}"
                    if np.isfinite(value)
                    else "binNA"
                )
                _accumulate(vector, self._value_tokens(name, label, label))
            else:
                raw = str(record[name])
                _accumulate(vector, self._value_tokens(name, raw, spec.decode(raw)))
        if str(record.get("text", "")):
            _accumulate(
                vector,
                [
                    f"text:{w}"
                    for w in _tokenize(record["text"])[: self.max_text_tokens]
                ],
            )
        norm = np.linalg.norm(vector)
        return (vector / max(norm, 1e-08)).astype(np.float32)


def build_semantic(
    frame: pd.DataFrame, train_idx: np.ndarray, gamma: SemanticSpec, cfg: Dict[str, Any]
) -> SemanticArtifacts:
    """Build semantic features according to the configured activation mode."""
    setting = str(cfg.get("enabled", "auto")).lower()
    has_text = (
        bool((frame["text"].astype(str).str.len() > 0).any())
        if "text" in frame.columns
        else False
    )
    if setting == "false":
        enabled, reason = (False, "Explicitly disabled")
    elif setting == "true":
        enabled, reason = (True, "Explicitly enabled")
    else:
        enabled = has_text
        reason = (
            "Raw text detected"
            if has_text
            else "No raw text; semantic features disabled in text-only mode"
        )
    encoder = DomainSemanticEncoder(gamma, cfg)
    visible, excluded = encoder._visible_columns(frame)
    if not enabled:
        return SemanticArtifacts(
            matrix=np.zeros((len(frame), 0), dtype=np.float32),
            enabled=False,
            reason=reason,
            excluded_by_omega=tuple(excluded),
            serialized_columns=tuple(visible),
            n_rows_checked=len(frame),
        )
    encoder.fit(frame, train_idx)
    guard = TemporalAvailabilityGuard(gamma)
    guard.assert_no_posthoc(encoder._columns)
    matrix = encoder.transform(frame)
    return SemanticArtifacts(
        matrix=matrix,
        enabled=True,
        reason=reason,
        excluded_by_omega=tuple(encoder._excluded),
        serialized_columns=tuple(encoder._columns),
        example_serialization=encoder.serialize_row(frame, int(train_idx[0])),
        n_rows_checked=len(frame),
    )
