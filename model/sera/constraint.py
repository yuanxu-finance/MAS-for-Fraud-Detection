from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
from scipy.stats import binom


class CalibrationInfeasible(RuntimeError):
    """Raised when the calibration sample cannot certify the requested risk budget."""


def minimum_negative_samples(alpha: float, delta: float) -> int:
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must be in (0,1)")
    return int(math.ceil(math.log(delta) / math.log1p(-alpha)))


def violation_probability(n0: int, k: int, alpha: float) -> float:
    """Return the binomial-tail bound for an order-statistic threshold."""
    if k <= 0:
        return 1.0
    if k > n0:
        return 0.0
    return float(binom.sf(k - 1, n0, 1.0 - alpha))


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    status: str
    alpha: float
    delta: float
    n_negative: int
    n_min: int
    k_star: Optional[int]
    threshold: float
    violation_bound: Optional[float]
    has_ties: bool
    scorer_version: int
    tie_admission_rate: float = 0.0
    n_tied_calibration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NPOrderStatisticCalibrator:

    def __init__(self, alpha: float, delta: float) -> None:
        if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
            raise ValueError("alpha and delta must be in (0,1)")
        self.alpha, self.delta = (alpha, delta)

    @property
    def n_min(self) -> int:
        return minimum_negative_samples(self.alpha, self.delta)

    def fit(
        self, negative_scores: Sequence[float], scorer_version: int
    ) -> CalibrationResult:
        scores = np.sort(np.asarray(negative_scores, dtype=np.float64))
        n0 = int(scores.size)
        if n0 < self.n_min:
            raise CalibrationInfeasible(
                f"n_0={n0} < n_0,min={self.n_min}(α={self.alpha}, δ={self.delta}): no certifiable threshold for this window"
            )
        if violation_probability(n0, n0, self.alpha) > self.delta:
            raise CalibrationInfeasible("No order statistic satisfies v(k) <= delta")
        low, high = (1, n0)
        while low < high:
            mid = (low + high) // 2
            if violation_probability(n0, mid, self.alpha) <= self.delta:
                high = mid
            else:
                low = mid + 1
        k_star = low
        threshold = float(scores[k_star - 1])
        n_tied = int(np.count_nonzero(scores == threshold))
        has_ties = n_tied > 1
        n_gt = int(np.count_nonzero(scores > threshold))
        m = max(0, n0 - k_star - n_gt)
        tie_rate = 0.0 if n_tied == 0 else min(1.0, m / n_tied)
        return CalibrationResult(
            method="np_order_statistic",
            status="CERTIFIED_WITH_TIES" if has_ties else "CERTIFIED",
            alpha=self.alpha,
            delta=self.delta,
            n_negative=n0,
            n_min=self.n_min,
            k_star=k_star,
            threshold=threshold,
            violation_bound=violation_probability(n0, k_star, self.alpha),
            has_ties=has_ties,
            scorer_version=scorer_version,
            tie_admission_rate=tie_rate,
            n_tied_calibration=n_tied,
        )


class EmpiricalThresholdCalibrator:
    """Select an empirical FPR threshold without an NP finite-sample guarantee."""

    def __init__(self, alpha: float, delta: float) -> None:
        self.alpha, self.delta = (alpha, delta)

    def fit(
        self, negative_scores: Sequence[float], scorer_version: int
    ) -> CalibrationResult:
        scores = np.sort(np.asarray(negative_scores, dtype=np.float64))
        n0 = int(scores.size)
        if n0 == 0:
            raise CalibrationInfeasible("Calibration set has no negative samples")
        k = int(math.ceil((1.0 - self.alpha) * n0))
        k = min(max(k, 1), n0)
        threshold = float(scores[k - 1])
        n_tied = int(np.count_nonzero(scores == threshold))
        n_gt = int(np.count_nonzero(scores > threshold))
        m = max(0, n0 - k - n_gt)
        tie_rate = 0.0 if n_tied == 0 else min(1.0, m / n_tied)
        return CalibrationResult(
            method="empirical_quantile",
            status="UNCERTIFIED",
            alpha=self.alpha,
            delta=self.delta,
            n_negative=n0,
            n_min=minimum_negative_samples(self.alpha, self.delta),
            k_star=k,
            threshold=threshold,
            violation_bound=violation_probability(n0, k, self.alpha),
            has_ties=bool(n_tied > 1),
            scorer_version=scorer_version,
            tie_admission_rate=tie_rate,
            n_tied_calibration=n_tied,
        )


@dataclass(frozen=True)
class CapacityResult:
    alerts: np.ndarray
    candidates: int
    emitted: int
    capacity_limited: bool
    capacity: Optional[int]


class CapacityGate:
    """Limit alert counts by removing alerts after thresholding."""

    def __init__(self, capacity: Optional[int]) -> None:
        if capacity is not None and capacity < 0:
            raise ValueError("capacity must be nonnegative or None")
        self.capacity = capacity

    def apply(
        self, scores: Sequence[float], threshold: float, tie_admission_rate: float = 0.0
    ) -> CapacityResult:
        """Admit the calibrated fraction of threshold ties in original row order."""
        s = np.asarray(scores, dtype=np.float64)
        candidate_idx = np.flatnonzero(s > threshold)
        if tie_admission_rate > 0.0:
            tied_idx = np.flatnonzero(s == threshold)
            take = int(np.floor(tie_admission_rate * tied_idx.size))
            if take > 0:
                candidate_idx = np.union1d(candidate_idx, tied_idx[:take])
        alerts = np.zeros(s.size, dtype=bool)
        if self.capacity is None or candidate_idx.size <= self.capacity:
            alerts[candidate_idx] = True
        elif self.capacity > 0:
            ranked = candidate_idx[np.argsort(s[candidate_idx])[::-1]]
            alerts[ranked[: self.capacity]] = True
        return CapacityResult(
            alerts,
            int(candidate_idx.size),
            int(alerts.sum()),
            bool(alerts.sum() < candidate_idx.size),
            self.capacity,
        )


class FixedPolicyCalibrator:
    """Use a fixed alert threshold without calibration."""

    def __init__(self, alpha: float, delta: float, investigate_at: float = 0.4) -> None:
        self.alpha, self.delta, self.investigate_at = (alpha, delta, investigate_at)

    def fit(
        self, negative_scores: Sequence[float], scorer_version: int
    ) -> CalibrationResult:
        n0 = int(np.asarray(negative_scores).size)
        return CalibrationResult(
            method="fixed_policy",
            status="FIXED_POLICY(no finite-sample guarantee)",
            alpha=self.alpha,
            delta=self.delta,
            n_negative=n0,
            n_min=minimum_negative_samples(self.alpha, self.delta),
            k_star=None,
            threshold=float(self.investigate_at),
            violation_bound=None,
            has_ties=False,
            scorer_version=scorer_version,
        )


def make_calibrator(method: str, alpha: float, delta: float):
    if method == "np":
        return NPOrderStatisticCalibrator(alpha, delta)
    if method == "empirical":
        return EmpiricalThresholdCalibrator(alpha, delta)
    if method == "fixed":
        return FixedPolicyCalibrator(alpha, delta)
    raise ValueError(f"Unknown calibration method {method!r}")
