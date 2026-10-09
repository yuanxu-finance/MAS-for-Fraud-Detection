from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np


class ContractViolation(RuntimeError):
    """Raised when a pipeline contract is violated."""


@dataclass(frozen=True)
class FieldSpec:
    """Field metadata, entity role, and decision-time availability."""

    name: str
    description: str = ""
    unit: str = ""
    entity_role: str = ""
    source: str = ""
    codebook: Mapping[str, str] = field(default_factory=dict)
    availability_lag_seconds: float = 0.0
    posthoc: bool = False
    in_serialization: bool = True

    def decode(self, value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "<missing>"
        return str(self.codebook.get(str(value), value))

    def available_at(self, event_time: float) -> float:
        if self.posthoc:
            return float("inf")
        return float(event_time) + float(self.availability_lag_seconds)


@dataclass(frozen=True)
class SemanticSpec:
    """Shared field schema for semantic, historical, and graph features."""

    fields: Mapping[str, FieldSpec]
    task_description: str
    version: str = "gamma-v1"

    @classmethod
    def from_config(cls, semantic_cfg: Mapping[str, Any]) -> "SemanticSpec":
        specs: Dict[str, FieldSpec] = {}
        for item in semantic_cfg.get("fields", []):
            specs[str(item["name"])] = FieldSpec(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                unit=str(item.get("unit", "")),
                entity_role=str(item.get("entity_role", "")),
                source=str(item.get("source", "")),
                codebook={
                    str(k): str(v) for k, v in (item.get("codebook") or {}).items()
                },
                availability_lag_seconds=float(
                    item.get("availability_lag_seconds", 0.0)
                ),
                posthoc=bool(item.get("posthoc", False)),
                in_serialization=bool(item.get("in_serialization", True)),
            )
        return cls(
            fields=specs,
            task_description=str(semantic_cfg.get("task_prompt", "")),
            version=str(semantic_cfg.get("version", "gamma-v1")),
        )

    def field(self, name: str) -> FieldSpec:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(f"Field {name!r} is not registered in the schema") from exc

    def is_empty_spec(self) -> bool:
        """Return whether the schema contains no field specifications."""
        return all(
            (
                not (f.description or f.unit or f.codebook or f.entity_role or f.source)
                for f in self.fields.values()
            )
        )

    def posthoc_fields(self) -> Tuple[str, ...]:
        return tuple(
            sorted((name for name, spec in self.fields.items() if spec.posthoc))
        )

    def validate_columns(self, columns: Iterable[str]) -> None:
        unknown = sorted(set(columns) - set(self.fields))
        if unknown:
            raise ContractViolation(
                f"Feature columns missing from the schema: {unknown}"
            )


class TemporalAvailabilityGuard:
    """Validate decision-time field, history, and edge availability."""

    def __init__(self, gamma: SemanticSpec) -> None:
        self.gamma = gamma

    def visible_fields(
        self, columns: Sequence[str], event_time: float, decision_time: float
    ) -> Tuple[List[str], List[str]]:
        """Return available fields and fields excluded by their availability times."""
        visible, excluded = ([], [])
        for name in columns:
            spec = self.gamma.field(name)
            if spec.available_at(event_time) <= decision_time:
                visible.append(name)
            else:
                excluded.append(name)
        return (visible, excluded)

    def assert_no_posthoc(self, used_columns: Iterable[str]) -> int:
        """Reject post-event fields and return the number of checked columns."""
        used = list(used_columns)
        banned = set(self.gamma.posthoc_fields())
        hit = sorted(set(used) & banned)
        if hit:
            raise ContractViolation(
                f"Post-event fields found in feature inputs: {hit} (availability time is infinite)"
            )
        return len(used)

    @staticmethod
    def assert_history_window(
        current_ts: np.ndarray,
        history_ts: np.ndarray,
        valid_mask: np.ndarray,
        window_seconds: float,
    ) -> int:
        """Validate admitted history lags and return the checked slot count."""
        if valid_mask.size == 0:
            return 0
        lag = current_ts[:, None] - history_ts
        bad = valid_mask & ~((lag > 0) & (lag <= window_seconds))
        if bool(bad.any()):
            idx = np.argwhere(bad)[0]
            raise ContractViolation(
                f"Admitted history outside the allowed window: row {int(idx[0])} slot {int(idx[1])} lag={float(lag[idx[0], idx[1]])}"
            )
        return int(valid_mask.size)

    @staticmethod
    def assert_causal_edges(edge_ts: np.ndarray, decision_time: float) -> int:
        """Reject future edges and return the number of checked edges."""
        if edge_ts.size and float(edge_ts.max()) > decision_time:
            raise ContractViolation(
                f"Future edge in the as-of view: max(edge_time)={float(edge_ts.max())} > t={decision_time}"
            )
        return int(edge_ts.size)


class ScorerLifecycle:
    """Require scorer freezing before calibration and deployment."""

    TRAINING, TUNING, FROZEN, CALIBRATED = (
        "TRAINING",
        "TUNING",
        "FROZEN",
        "CALIBRATED",
    )

    def __init__(self) -> None:
        self.state = self.TRAINING
        self.version = 0

    def enter_tuning(self) -> None:
        if self.state != self.TRAINING:
            raise ContractViolation(f"Cannot transition from {self.state} to tuning")
        self.state = self.TUNING

    def freeze(self) -> int:
        if self.state not in (self.TRAINING, self.TUNING):
            raise ContractViolation(f"Cannot transition from {self.state} to frozen")
        self.version += 1
        self.state = self.FROZEN
        return self.version

    def mark_calibrated(self) -> None:
        if self.state != self.FROZEN:
            raise ContractViolation("NP calibration requires a frozen scorer")
        self.state = self.CALIBRATED

    def assert_scoring_allowed(self) -> None:
        if self.state not in (self.FROZEN, self.CALIBRATED):
            raise ContractViolation("Deployment scoring requires a frozen scorer")


@dataclass
class CheckResult:
    name: str
    status: str
    n_checked: int
    detail: str = ""


class ContractAuditor:
    """Record check counts and statuses; empty checks are marked SKIPPED."""

    CONTROL_LAYER_NAMES = (
        "trigger",
        "triggered",
        "is_trigger",
        "m_t",
        "epsilon_t",
        "burst_gate",
        "seed_flag",
    )

    def __init__(self) -> None:
        self.results: List[CheckResult] = []

    def record(
        self, name: str, n_checked: int, ok: bool = True, detail: str = ""
    ) -> None:
        if n_checked <= 0:
            self.results.append(
                CheckResult(name, "SKIPPED", 0, detail or "No objects to check")
            )
        else:
            self.results.append(
                CheckResult(name, "PASSED" if ok else "FAILED", n_checked, detail)
            )

    def flag(self, name: str, n_checked: int, clean: bool, detail: str = "") -> None:
        """Record diagnostic exceedances as REVIEW rather than contract failures."""
        if n_checked <= 0:
            self.results.append(
                CheckResult(name, "SKIPPED", 0, detail or "No objects to check")
            )
        else:
            self.results.append(
                CheckResult(name, "PASSED" if clean else "REVIEW", n_checked, detail)
            )

    def run(self, name: str, fn, detail: str = "") -> None:
        """Run a check whose return value is the number of objects inspected."""
        try:
            n = int(fn())
            self.record(name, n, True, detail)
        except (ContractViolation, ValueError, KeyError) as exc:
            self.results.append(
                CheckResult(name, "FAILED", -1, f"{type(exc).__name__}: {exc}")
            )

    def assert_control_not_in_scorer(self, feature_names: Sequence[str]) -> int:
        lowered = {n.lower() for n in feature_names}
        hit = sorted((n for n in self.CONTROL_LAYER_NAMES if n in lowered))
        if hit:
            raise ContractViolation(
                f"Control-layer quantities found in scorer inputs: {hit}"
            )
        return len(feature_names)

    @property
    def passed(self) -> bool:
        return all((r.status != "FAILED" for r in self.results))

    @property
    def n_skipped(self) -> int:
        return sum((1 for r in self.results if r.status == "SKIPPED"))

    @property
    def n_review(self) -> int:
        return sum((1 for r in self.results if r.status == "REVIEW"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.results),
            "n_passed": sum((1 for r in self.results if r.status == "PASSED")),
            "n_failed": sum((1 for r in self.results if r.status == "FAILED")),
            "n_review": self.n_review,
            "n_skipped": self.n_skipped,
            "checks": [
                {
                    "name": r.name,
                    "status": r.status,
                    "n_checked": r.n_checked,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }

    def summary_line(self) -> str:
        return f"Contract audit: {sum((1 for r in self.results if r.status == 'PASSED'))} PASSED / {sum((1 for r in self.results if r.status == 'FAILED'))} FAILED / {self.n_review} REVIEW / {self.n_skipped} SKIPPED"


@dataclass(frozen=True)
class PatternEntry:
    """Structural, relational, temporal, and attribute evidence for a pattern."""

    pattern_id: str
    S: Tuple[str, ...]
    R: Tuple[Tuple[str, str, str], ...]
    T: Tuple[float, ...]
    A: Mapping[str, Any]
    seed_pair: Tuple[str, str]
    time_bucket: int
    burst_score: float
    concentration: float
    zeta: float
    epsilon_t: float
    m_t: int
    novelty: str = "UNKNOWN"
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "S": list(self.S),
            "R": [list(r) for r in self.R],
            "T": [float(t) for t in self.T],
            "A": dict(self.A),
            "seed_pair": list(self.seed_pair),
            "time_bucket": int(self.time_bucket),
            "burst_score": float(self.burst_score),
            "concentration": float(self.concentration),
            "zeta": float(self.zeta),
            "epsilon_t": float(self.epsilon_t),
            "m_t": int(self.m_t),
            "novelty": self.novelty,
            "signature": self.signature,
        }
