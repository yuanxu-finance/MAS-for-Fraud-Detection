from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(average_precision_score(labels, scores))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Evaluate TPR at the negative-score quantile for the requested FPR."""
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if negatives.size == 0 or positives.size == 0:
        return float("nan")
    threshold = float(np.quantile(negatives, 1.0 - target_fpr, method="higher"))
    return float((positives > threshold).mean())


def clopper_pearson_upper(false_positives: int, negatives: int, delta: float) -> float:
    if negatives <= 0:
        return 1.0
    if false_positives >= negatives:
        return 1.0
    return float(
        beta.ppf(1.0 - delta, false_positives + 1, negatives - false_positives)
    )


@dataclass(frozen=True)
class EvaluationResult:
    n: int
    n_positive: int
    n_negative: int
    prevalence: float
    ap: float
    ap_lift: float
    auroc: float
    tpr_at_fpr: Dict[str, float]
    threshold: float
    n_alert: int
    n_tp: int
    n_fp: int
    tpr: float
    fpr: float
    precision: float
    fdp: float
    fpr_upper_bound: float
    constraint_satisfied: bool
    n_tn: int = 0
    n_fn: int = 0
    f1: float = 0.0
    accuracy: float = 0.0
    ap_ci_low: float = float("nan")
    ap_ci_high: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def evaluate(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
    alpha: float,
    delta: float,
    alerts: Optional[Sequence[bool]] = None,
    fpr_grid: Sequence[float] = (0.001, 0.005, 0.01),
    bootstrap: int = 0,
    seed: int = 2026,
) -> EvaluationResult:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    predicted = np.asarray(alerts, dtype=bool) if alerts is not None else s > threshold
    pos, neg = (y == 1, y == 0)
    n_pos, n_neg = (int(pos.sum()), int(neg.sum()))
    tp, fp = (int((predicted & pos).sum()), int((predicted & neg).sum()))
    tn, fn = (n_neg - fp, n_pos - tp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, n_pos)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, y.size)
    prevalence = _safe_div(n_pos, y.size)
    ap = average_precision(y, s)
    upper = clopper_pearson_upper(fp, n_neg, delta)
    ci_low = ci_high = float("nan")
    if bootstrap > 0 and n_pos > 0:
        draws = bootstrap_ap(y, s, repeats=bootstrap, seed=seed)
        ci_low, ci_high = (
            float(np.nanpercentile(draws, 2.5)),
            float(np.nanpercentile(draws, 97.5)),
        )
    return EvaluationResult(
        n=int(y.size),
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=prevalence,
        ap=ap,
        ap_lift=_safe_div(ap, prevalence) if prevalence else float("nan"),
        auroc=roc_auc(y, s),
        tpr_at_fpr={f"{f:g}": tpr_at_fpr(y, s, f) for f in fpr_grid},
        threshold=float(threshold),
        n_alert=int(predicted.sum()),
        n_tp=tp,
        n_fp=fp,
        tpr=recall,
        fpr=_safe_div(fp, n_neg),
        precision=precision,
        fdp=1.0 - precision,
        fpr_upper_bound=upper,
        constraint_satisfied=bool(upper <= alpha),
        n_tn=tn,
        n_fn=fn,
        f1=f1,
        accuracy=accuracy,
        ap_ci_low=ci_low,
        ap_ci_high=ci_high,
    )


def bootstrap_ap(
    labels: np.ndarray, scores: np.ndarray, repeats: int = 1000, seed: int = 2026
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = labels.size
    out = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        idx = rng.integers(0, n, n)
        y, s = (labels[idx], scores[idx])
        out[i] = average_precision(y, s) if 0 < y.sum() < y.size else np.nan
    return out


def paired_bootstrap_delta(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    repeats: int = 1000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Estimate paired metric differences using shared bootstrap indices."""
    rng = np.random.default_rng(seed)
    n = labels.size
    deltas = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if not 0 < y.sum() < y.size:
            deltas[i] = np.nan
            continue
        deltas[i] = average_precision(y, scores_a[idx]) - average_precision(
            y, scores_b[idx]
        )
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_two_sided": float("nan"),
            "n_draws": 0,
        }
    n = float(finite.size)
    left = (float((finite <= 0).sum()) + 1.0) / (n + 1.0)
    right = (float((finite >= 0).sum()) + 1.0) / (n + 1.0)
    p = 2.0 * min(left, right)
    return {
        "delta_mean": float(finite.mean()),
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_two_sided": float(min(1.0, p)),
        "n_draws": int(finite.size),
    }
