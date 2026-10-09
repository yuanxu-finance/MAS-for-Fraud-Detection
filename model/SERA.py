"""SERA evidence construction, fusion scoring, and calibrated evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import binom, chi2
from torch import nn

from training.config import (
    DATASETS,
    config_digest,
    get_path,
    load_model_config as load_config,
)
from training.datasets import Workspace
from training.metrics import average_precision, evaluate, tpr_at_fpr


class SERA:
    """Construct evidence, fit a fusion scorer, and calibrate independent alerts."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = copy.deepcopy(dict(config))

    def fit_evaluate(
        self,
        ws: Workspace,
        *,
        arm: str = "full",
        verbose: bool = True,
        score_transform=None,
    ) -> Dict[str, Any]:
        """Fit on training/tuning splits, calibrate negatives, and evaluate the test split."""
        cfg = self.config
        seed = int(cfg.get("seed", 2026))
        frame, splits, y = (ws.frame, ws.splits, ws.y)
        gamma = SemanticSpec.from_config(cfg["semantic"])
        auditor = ContractAuditor()
        timings: Dict[str, float] = {}
        auditor.run(
            "Four disjoint, chronologically ordered segments",
            lambda: splits.assert_valid(frame["ts"].to_numpy()),
            f"mode={splits.mode}, sizes={splits.sizes()}",
        )
        t0 = time.perf_counter()
        context = _cached(
            ws,
            "context",
            cfg["context"],
            lambda a: build_context(
                frame, splits.train, y, gamma, cfg["context"], seed, a
            ),
            auditor,
        )
        timings["context"] = time.perf_counter() - t0
        graph_key = {k: v for k, v in cfg["graph"].items() if k not in _RHO_KEYS}
        t0 = time.perf_counter()
        graph_base = _cached(
            ws,
            "graph",
            graph_key,
            lambda a: build_graph(frame, splits.train, gamma, cfg["graph"], seed, a),
            auditor,
        )
        timings["graph"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        selection, semantic, graph, X, names, detector, det_artifacts = _select_on_tune(
            ws, cfg, gamma, context, graph_base, auditor, seed
        )
        timings["selection"] = time.perf_counter() - t0
        context_selected = bool(
            selection.get("chosen_context", context.matrix.shape[1] > 0)
        )
        auditor.record(
            "Semantic inputs exclude post-event fields",
            len(semantic.serialized_columns),
            True,
            f"Serialized columns: {list(semantic.serialized_columns)}; excluded by availability time: {list(semantic.excluded_by_omega) or 'none'}",
        )
        auditor.run(
            "Scorer inputs exclude control-layer quantities",
            lambda: auditor.assert_control_not_in_scorer(names),
            "Screening gates and test parameters are excluded from risk features",
        )
        if graph.enabled:
            gs = graph.stats
            auditor.record(
                "Confirmations at the selected concentration threshold",
                int(gs.get("n_expanded", 0)),
                True,
                f"threshold={gs.get('rho_min_used')}; expanded {gs.get('n_expanded')} -> confirmed {gs.get('n_structure_confirmed')} (temporal-only {gs.get('n_time_only')}, no path {gs.get('n_no_path')}); transactions with nonzero graph features: {gs.get('n_rows_with_signal')}",
            )
        auditor.record(
            "Selection uses training and tuning segments only",
            int(selection["n_candidates"]),
            True,
            f"criterion={selection['criterion']}; selected semantic={selection['chosen_semantic']}, rho_quantile={selection['chosen_rho_quantile']}; calibration and test segments excluded",
        )
        probe = leakage_probe(X, y, names, splits.train, splits.test)
        auditor.flag(
            "Feature-label correlation diagnostic on out-of-time test data",
            int(probe.get("n_features", 0)),
            not probe.get("flagged_on_test"),
            f"test max|r|={probe.get('max_abs_r_test', float('nan')):.4f} (train: {probe.get('max_abs_r_train', float('nan')):.4f}); flagged test columns: {probe.get('flagged_on_test') or 'none'}",
        )
        lifecycle = ScorerLifecycle()
        lifecycle.enter_tuning()
        scorer_version = lifecycle.freeze()
        auditor.record(
            "Scorer frozen before calibration",
            1,
            True,
            f"state={lifecycle.state}, version={scorer_version}, best_iteration={det_artifacts.best_iteration}",
        )
        alpha = float(get_path(cfg, "constraint.alpha"))
        delta = float(get_path(cfg, "constraint.delta"))
        method = str(get_path(cfg, "constraint.method"))
        calibrator = make_calibrator(method, alpha, delta)
        cal_scores = detector.predict(X[splits.calibration])
        if score_transform is not None:
            cal_scores = score_transform(cal_scores, "calibration")
        cal_negatives = cal_scores[y[splits.calibration] == 0]
        calibration_error: Optional[str] = None
        try:
            calibration = calibrator.fit(cal_negatives, scorer_version)
            lifecycle.mark_calibrated()
            calibration_dict = calibration.to_dict()
            threshold = calibration.threshold
        except CalibrationInfeasible as exc:
            calibration_error = str(exc)
            calibration_dict = {
                "method": method,
                "status": "INFEASIBLE",
                "detail": str(exc),
                "n_negative": int(cal_negatives.size),
            }
            threshold = float("inf")
        n_min = getattr(calibrator, "n_min", "n/a")
        auditor.record(
            "Independent calibration samples excluded from training and selection",
            int(cal_negatives.size),
            calibration_error is None,
            calibration_error or f"n_0={cal_negatives.size}, n_0,min={n_min}",
        )
        t0 = time.perf_counter()
        test_scores = detector.predict(X[splits.test])
        if score_transform is not None:
            test_scores = score_transform(test_scores, "test")
        timings["score_test"] = time.perf_counter() - t0
        y_test = y[splits.test]
        capacity = CapacityGate(get_path(cfg, "constraint.capacity_per_window"))
        gate = capacity.apply(
            test_scores,
            threshold,
            tie_admission_rate=float(calibration_dict.get("tie_admission_rate", 0.0)),
        )
        metrics = evaluate(
            y_test,
            test_scores,
            threshold=threshold,
            alpha=alpha,
            delta=delta,
            alerts=gate.alerts,
            fpr_grid=list(get_path(cfg, "evaluation.fpr_grid")),
            bootstrap=int(get_path(cfg, "evaluation.bootstrap", 0)),
            seed=seed,
        )
        t0 = time.perf_counter()
        explanations = _explain(
            detector,
            X[splits.test],
            gate.alerts,
            int(get_path(cfg, "evaluation.shap_examples", 20)),
            int(get_path(cfg, "evaluation.shap_top_k", 3)),
        )
        timings["shap"] = time.perf_counter() - t0
        novelty = graph.stats.get("novelty", {}) if graph.enabled else {}
        n_added = (
            (int(semantic.matrix.shape[1]) if semantic.enabled else 0)
            + (int(context.matrix.shape[1]) if context_selected else 0)
            + (int(graph.matrix.shape[1]) if graph.enabled else 0)
        )
        result: Dict[str, Any] = {
            "degenerate_to_x_base": bool(n_added == 0),
            "arm": arm,
            "dataset": ws.dataset.name,
            "seed": seed,
            "config_digest": config_digest(cfg),
            "dataset_meta": ws.dataset.meta,
            "split_sizes": splits.sizes(),
            "split_mode": splits.mode,
            "selection_on_tune": selection,
            "layers": {
                "L1_semantic": {
                    "enabled": semantic.enabled,
                    "reason": semantic.reason,
                    "dim": int(semantic.matrix.shape[1]),
                    "excluded_by_omega": list(semantic.excluded_by_omega),
                    "serialized_columns": list(semantic.serialized_columns),
                    "example": semantic.example_serialization[:600],
                },
                "L1_context": {
                    "enabled": bool(context_selected),
                    "reason": (
                        context.reason
                        if context_selected
                        else f"Disabled by tuning selection ({context.reason})"
                    ),
                    "mode": context.mode,
                    "negative_control": context.negative_control,
                    "dim": int(context.matrix.shape[1]) if context_selected else 0,
                    "admission": context.admission,
                    "training": context.training,
                },
                "L2_graph": {
                    "enabled": graph.enabled,
                    "reason": graph.reason,
                    "dim": int(graph.matrix.shape[1]),
                    "stats": graph.stats,
                    "n_patterns_exported": len(graph.patterns),
                    "novelty": novelty,
                },
                "L3_constraint": {
                    "method": method,
                    "alpha": alpha,
                    "delta": delta,
                    "capacity_per_window": get_path(
                        cfg, "constraint.capacity_per_window"
                    ),
                },
            },
            "detector": {
                "best_iteration": det_artifacts.best_iteration,
                "best_score_aucpr_tune": det_artifacts.best_score,
                "n_features": det_artifacts.n_features,
                "params": det_artifacts.params,
                "top_gain": det_artifacts.global_importance[:15],
            },
            "calibration": calibration_dict,
            "capacity_gate": {
                "candidates": gate.candidates,
                "emitted": gate.emitted,
                "capacity_limited": gate.capacity_limited,
                "capacity": gate.capacity,
            },
            "test_metrics": metrics.to_dict(),
            "leakage_probe": probe,
            "contracts": auditor.to_dict(),
            "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
            "explanations": explanations,
            "patterns_preview": [p.to_dict() for p in graph.patterns[:10]],
        }
        result["_scores"] = test_scores
        result["_calibration_scores"] = cal_scores
        result["_tune_scores"] = detector.predict(X[splits.tune])
        result["_patterns"] = graph.patterns
        if verbose:
            _print_arm(result)
        return result


def _cached(
    ws: Workspace,
    layer: str,
    section: Mapping[str, Any],
    builder: Callable[[ContractAuditor], Any],
    auditor: ContractAuditor,
) -> Any:
    """Cache layer outputs together with their contract-check records."""
    digest = hashlib.blake2b(
        json.dumps(section, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        ),
        digest_size=8,
    ).hexdigest()
    entry = ws.cache.get((layer, digest))
    if entry is None:
        local = ContractAuditor()
        entry = (builder(local), list(local.results))
        ws.cache[layer, digest] = entry
    auditor.results.extend(entry[1])
    return entry[0]


_RHO_KEYS = (
    "rho_min",
    "rho_quantile",
    "rho_quantile_grid",
    "rho_fallback",
    "rho_min_calibration_seeds",
    "zeta_mode",
    "zeta_mode_grid",
)
_TRUTHY = ("true", "1", "yes", "on")


def _as_bool(setting: str) -> bool:
    return setting in _TRUTHY


def _semantic_candidates(cfg: Mapping[str, Any], frame: pd.DataFrame) -> List[bool]:
    """Resolve semantic activation candidates for tuning-set selection."""
    setting = str(cfg.get("enabled", "auto")).lower()
    if setting == "auto":
        return [True, False]
    if setting == "text_only":
        has_text = (
            bool((frame["text"].astype(str).str.len() > 0).any())
            if "text" in frame
            else False
        )
        return [has_text]
    return [_as_bool(setting)]


def _context_candidates(cfg: Mapping[str, Any]) -> List[bool]:
    """Resolve history activation candidates for tuning-set selection."""
    setting = str(cfg.get("enabled", True)).lower()
    if setting == "auto":
        return [True, False]
    return [_as_bool(setting)]


def _rho_candidates(
    cfg: Mapping[str, Any], graph_enabled: bool
) -> List[Optional[float]]:
    if not graph_enabled:
        return [None]
    setting = cfg.get("rho_quantile", 0.75)
    if isinstance(setting, str) and setting.lower() == "auto":
        return [float(q) for q in cfg.get("rho_quantile_grid", [0.0, 0.5, 0.75])]
    return [float(setting)]


def _zeta_candidates(
    cfg: Mapping[str, Any], graph_enabled: bool
) -> List[Optional[str]]:
    """Resolve intensity-weighting candidates independently of concentration gates."""
    if not graph_enabled:
        return [None]
    setting = cfg.get("zeta_mode", "product")
    if isinstance(setting, str) and setting.lower() == "auto":
        return list(cfg.get("zeta_mode_grid", ["product", "burst_only"]))
    return [str(setting)]


def _select_on_tune(
    ws: Workspace,
    cfg: Dict[str, Any],
    gamma: SemanticSpec,
    context: ContextArtifacts,
    graph_base: GraphArtifacts,
    auditor: ContractAuditor,
    seed: int,
) -> Tuple[Any, ...]:
    """Select modules and graph settings on tuning data and retain the fitted scorer."""
    frame, splits, y = (ws.frame, ws.splits, ws.y)
    criterion = str(get_path(cfg, "evaluation.selection_criterion", "ap"))
    alpha = float(get_path(cfg, "constraint.alpha"))
    y_tune = y[splits.tune]
    empty_ctx = np.zeros((ws.X_base.shape[0], 0), dtype=np.float32)
    ctx_candidates = (
        _context_candidates(cfg["context"]) if context.matrix.shape[1] else [False]
    )
    table: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for use_semantic in _semantic_candidates(cfg["semantic"], frame):
        semantic_cfg = dict(cfg["semantic"])
        semantic_cfg["enabled"] = bool(use_semantic)
        semantic = _cached(
            ws,
            "semantic",
            semantic_cfg,
            lambda a, c=semantic_cfg: build_semantic(frame, splits.train, gamma, c),
            auditor,
        )
        semantic_names = [f"h_dom_{i}" for i in range(semantic.matrix.shape[1])]
        for use_context in ctx_candidates:
            ctx_matrix = context.matrix if use_context else empty_ctx
            ctx_names = context.feature_names if use_context else []
            for quantile in _rho_candidates(cfg["graph"], graph_base.enabled):
                for zeta_mode in _zeta_candidates(cfg["graph"], graph_base.enabled):
                    if quantile is None:
                        graph = graph_base
                    else:
                        threshold, calibration = graph_base.resolve_threshold(
                            cfg["graph"].get("rho_min", "auto"),
                            quantile,
                            int(cfg["graph"].get("rho_min_calibration_seeds", 20)),
                            float(cfg["graph"].get("rho_fallback", 0.2)),
                        )
                        graph = graph_base.with_threshold(
                            threshold, calibration, zeta_mode
                        )
                    X, names = _stack(
                        [
                            (ws.X_base, ws.base_names),
                            (semantic.matrix, semantic_names),
                            (ctx_matrix, ctx_names),
                            (graph.matrix, graph.feature_names),
                        ]
                    )
                    detector = FusedDetector(cfg["detector"], names, seed)
                    artifacts = detector.fit(
                        X[splits.train], y[splits.train], X[splits.tune], y[splits.tune]
                    )
                    tune_scores = detector.predict(X[splits.tune])
                    score_ap = average_precision(y_tune, tune_scores)
                    score_tpr = tpr_at_fpr(y_tune, tune_scores, alpha)
                    value = score_tpr if criterion == "tpr_at_alpha" else score_ap
                    row = {
                        "semantic": bool(use_semantic),
                        "context": bool(use_context),
                        "rho_quantile": quantile,
                        "zeta_mode": zeta_mode,
                        "tune_ap": score_ap,
                        "tune_tpr_at_alpha": score_tpr,
                        "n_features": int(X.shape[1]),
                        "graph_rows_with_signal": int(
                            graph.stats.get("n_rows_with_signal", 0)
                        ),
                    }
                    table.append(row)
                    if best is None or value > best["value"]:
                        best = {
                            "value": value,
                            "row": row,
                            "semantic": semantic,
                            "graph": graph,
                            "X": X,
                            "names": names,
                            "detector": detector,
                            "artifacts": artifacts,
                        }
    if best is None:
        raise RuntimeError("Tuning produced no candidate scorer")
    for row in table:
        row["selected"] = row is best["row"]
    value_key = "tune_tpr_at_alpha" if criterion == "tpr_at_alpha" else "tune_ap"
    values = sorted((r[value_key] for r in table), reverse=True)
    margin = values[0] - values[1] if len(values) > 1 else float("nan")
    n_tied = sum((1 for v in values if v == values[0]))
    selection = {
        "criterion": criterion,
        "n_candidates": len(table),
        "table": table,
        "chosen_semantic": best["row"]["semantic"],
        "chosen_context": best["row"].get("context"),
        "chosen_rho_quantile": best["row"]["rho_quantile"],
        "chosen_zeta_mode": best["row"].get("zeta_mode"),
        "chosen_tune_ap": best["row"]["tune_ap"],
        "chosen_tune_tpr_at_alpha": best["row"]["tune_tpr_at_alpha"],
        "margin_over_runner_up": margin,
        "n_tied_at_best": n_tied,
        "tie_broken_by_grid_order": bool(n_tied > 1),
    }
    return (
        selection,
        best["semantic"],
        best["graph"],
        best["X"],
        best["names"],
        best["detector"],
        best["artifacts"],
    )


def _stack(
    blocks: Sequence[Tuple[np.ndarray, Sequence[str]]]
) -> Tuple[np.ndarray, List[str]]:
    matrices = [m for m, names in blocks if m.size and m.shape[1] > 0]
    names: List[str] = []
    for _, block_names in blocks:
        names.extend(block_names)
    if not matrices:
        raise ValueError("Scorer has no input features")
    stacked = np.hstack(matrices).astype(np.float32)
    if stacked.shape[1] != len(names):
        raise ValueError(
            f"Scorer column count {stacked.shape[1]} differs from feature-name count {len(names)}: {names}"
        )
    return (stacked, names)


def _pearson_against_label(
    X: np.ndarray, y: np.ndarray, idx: np.ndarray
) -> Optional[np.ndarray]:
    sub = np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    target = y[idx].astype(np.float64)
    centered_target = target - target.mean()
    target_norm = float(np.sqrt((centered_target**2).sum()))
    if target_norm == 0.0:
        return None
    centered = sub - sub.mean(axis=0, keepdims=True)
    norms = np.sqrt((centered**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (centered * centered_target[:, None]).sum(axis=0) / (norms * target_norm)
    return np.nan_to_num(r, nan=0.0)


def leakage_probe(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    flag_threshold: float = 0.5,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Report feature-label correlations; test-set exceedances trigger review, not proof of leakage."""
    r_train = _pearson_against_label(X, y, train_idx)
    r_test = _pearson_against_label(X, y, test_idx)
    if r_train is None or r_test is None:
        return {
            "n_features": len(names),
            "note": "Correlation is undefined for a single-class segment",
        }
    order = np.argsort(np.abs(r_test))[::-1][:top_k]
    flagged = [names[int(i)] for i in np.flatnonzero(np.abs(r_test) > flag_threshold)]
    return {
        "n_features": len(names),
        "n_rows_train": int(train_idx.size),
        "n_rows_test": int(test_idx.size),
        "flag_threshold": flag_threshold,
        "flagged_on_test": flagged,
        "max_abs_r_train": float(np.abs(r_train).max()),
        "max_abs_r_test": float(np.abs(r_test).max()),
        "top_by_test": [
            {
                "feature": names[int(i)],
                "r_test": float(r_test[int(i)]),
                "r_train": float(r_train[int(i)]),
            }
            for i in order
        ],
    }


def _explain(
    detector: FusedDetector,
    X_test: np.ndarray,
    alerts: np.ndarray,
    n_examples: int,
    top_k: int,
) -> Dict[str, Any]:
    """Compute feature attributions for alerted test transactions."""
    idx = np.flatnonzero(alerts)[:n_examples]
    if idx.size == 0:
        return {
            "n_alerts_explained": 0,
            "examples": [],
            "note": "No test alerts to explain",
        }
    tops = detector.explain_top_k(X_test[idx], top_k)
    return {
        "n_alerts_explained": int(idx.size),
        "examples": [
            {"test_row": int(r), "top_features": t} for r, t in zip(idx, tops)
        ],
    }


def _print_arm(result: Dict[str, Any]) -> None:
    m = result["test_metrics"]
    layers = result["layers"]
    n_pos = int(m.get("n_positive", 0))
    print(
        f"  [Configuration] {result['arm']:<28} P={m['precision']:.4f} R={m['tpr']:.4f} AUC-ROC={m['auroc']:.4f} Acc={m['accuracy']:.6f}"
    )
    print(
        f"        AP={m['ap']:.5f} (×{m['ap_lift']:.1f} base rate)  TPR@FPR=1%={m['tpr_at_fpr'].get('0.01', float('nan')):.4f}  positives {n_pos}"
    )
    print(
        f"        L1 semantic {('ON ' if layers['L1_semantic']['enabled'] else 'OFF')}(d={layers['L1_semantic']['dim']}) | L1 context {('ON ' if layers['L1_context']['enabled'] else 'OFF')}(d={layers['L1_context']['dim']}) | L2 graph {('ON ' if layers['L2_graph']['enabled'] else 'OFF')}(d={layers['L2_graph']['dim']}) | L3 {result['calibration'].get('status')}"
    )
    sel = result.get("selection_on_tune", {})
    if sel.get("n_candidates", 0) > 1:
        margin = sel.get("margin_over_runner_up")
        tied = sel.get("n_tied_at_best") or 1
        quality = (
            "%d tied candidates; first in iteration order selected" % tied
            if tied > 1
            else "margin %.5f" % (margin if margin is not None else float("nan"))
        )
        print(
            f"        Tuning selection ({sel['criterion']}, {sel['n_candidates']} candidates, {quality}): semantic={sel['chosen_semantic']}, concentration quantile={sel['chosen_rho_quantile']}, ζ={sel.get('chosen_zeta_mode')}, tune AP={sel['chosen_tune_ap']:.5f}"
        )
    print(
        f"        threshold T={m['threshold']:.6g}  alerts {m['n_alert']}  TP={m['n_tp']} FP={m['n_fp']}  observed FPR={m['fpr']:.6f}  U_FP={m['fpr_upper_bound']:.6f}  constraint {('PASSED' if m['constraint_satisfied'] else 'FAILED')}"
    )
    contracts = result["contracts"]
    print(
        f"        {('Contract audit PASSED' if contracts['passed'] else 'Contract audit FAILED')}: {contracts['n_passed']} PASSED / {contracts['n_failed']} FAILED / {contracts.get('n_review', 0)} REVIEW / {contracts['n_skipped']} SKIPPED"
    )


def run(
    dataset="banksim", seed=2026, datasets_dir=None, output_dir=None, window_rows=None
):
    """Run the dataset workflow through the training entry point."""
    from training.pipeline import run as run_pipeline

    return run_pipeline(dataset, seed, datasets_dir, output_dir, window_rows)


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


AGG_NAMES = (
    "ctx_wmean_amount",
    "ctx_wdev_amount",
    "ctx_wmax_amount",
    "ctx_wmean_gap",
    "ctx_wmax_gap",
    "ctx_effective_count",
    "ctx_admitted_fraction",
    "ctx_recency_log",
)


@dataclass
class ContextArtifacts:
    matrix: np.ndarray
    feature_names: List[str]
    enabled: bool
    reason: str
    mode: str
    negative_control: str
    admission: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    n_window_checked: int = 0


def build_history_index(entity_code: np.ndarray, ts: np.ndarray, k: int) -> np.ndarray:
    """Retrieve up to k previous same-entity row indices, padding with -1."""
    order = np.lexsort((ts, entity_code))
    ent_sorted = entity_code[order]
    n = order.size
    sorted_hist = np.full((n, k), -1, dtype=np.int64)
    positions = np.arange(n)
    for lag in range(1, k + 1):
        src = positions - lag
        ok = src >= 0
        same = np.zeros(n, dtype=bool)
        same[ok] = ent_sorted[src[ok]] == ent_sorted[ok]
        sel = ok & same
        sorted_hist[sel, lag - 1] = order[src[sel]]
    out = np.empty_like(sorted_hist)
    out[order] = sorted_hist
    return out


def build_random_history_index(ts: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Sample strictly past cross-entity records as a history control."""
    order = np.argsort(ts, kind="mergesort")
    n = order.size
    rng = np.random.default_rng(seed)
    positions = np.arange(n)
    draws = rng.random((n, k))
    picked = np.floor(draws * positions[:, None]).astype(np.int64)
    picked = np.where(positions[:, None] > 0, picked, -1)
    sorted_hist = np.where(picked >= 0, order[np.clip(picked, 0, None)], -1)
    out = np.empty_like(sorted_hist)
    out[order] = sorted_hist
    return out


class HistoryAdmission:
    """Enforce temporal, field-availability, and business-compatibility constraints."""

    REASONS = (
        "no_history_slot",
        "outside_observation_window",
        "missing_required_field",
        "incompatible_measurement",
        "blocked_business_pair",
    )

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.window = float(cfg["window_seconds"])
        self.required_fields = list(cfg.get("required_fields", ["amount"]))
        self.compatibility_fields = list(cfg.get("compatibility_fields", []))
        self.blocked_value_pairs = {
            str(k): {tuple(map(str, p)) for p in v or []}
            for k, v in (cfg.get("blocked_value_pairs") or {}).items()
        }

    def evaluate(
        self, frame: pd.DataFrame, hist: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        n, k = hist.shape
        ts = frame["ts"].to_numpy(dtype=np.float64)
        has_slot = hist >= 0
        safe = np.clip(hist, 0, None)
        counts: Dict[str, int] = {r: 0 for r in self.REASONS}
        counts["no_history_slot"] = int((~has_slot).sum())
        lag = ts[:, None] - ts[safe]
        in_window = (lag > 0) & (lag <= self.window)
        mask = has_slot & in_window
        counts["outside_observation_window"] = int((has_slot & ~in_window).sum())
        for name in self.required_fields:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                dtype=np.float64
            )
            present = np.isfinite(values)
            ok = present[:, None] & present[safe]
            counts["missing_required_field"] = counts.get(
                "missing_required_field", 0
            ) + int((mask & ~ok).sum())
            mask &= ok
        for name in self.compatibility_fields:
            codes = pd.factorize(frame[name].astype(str))[0]
            ok = codes[:, None] == codes[safe]
            counts["incompatible_measurement"] = counts.get(
                "incompatible_measurement", 0
            ) + int((mask & ~ok).sum())
            mask &= ok
        blocked_total = 0
        for name, pairs in self.blocked_value_pairs.items():
            values = frame[name].astype(str).to_numpy()
            bad = np.zeros_like(mask)
            for row, col in zip(*np.nonzero(mask)):
                if (values[row], values[safe[row, col]]) in pairs:
                    bad[row, col] = True
            blocked_total += int(bad.sum())
            mask &= ~bad
        counts["blocked_business_pair"] = blocked_total
        counts["admitted"] = int(mask.sum())
        counts["slots_total"] = int(mask.size)
        counts["rows_with_history"] = int(mask.any(axis=1).sum())
        return (mask, counts)


def _hash_codes(values: np.ndarray, field_name: str, buckets: int) -> np.ndarray:
    uniq, inverse = np.unique(values.astype(str), return_inverse=True)
    table = np.empty(uniq.size, dtype=np.int64)
    for i, value in enumerate(uniq):
        digest = hashlib.blake2b(
            f"{field_name}::{value}".encode("utf-8"), digest_size=8
        ).digest()
        table[i] = int.from_bytes(digest, "little") % buckets
    return table[inverse].astype(np.int32)


class NAGContextModel(nn.Module):

    def __init__(
        self, n_fields: int, buckets: int, emb_dim: int, out_dim: int, mode: str
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_fields = n_fields
        self.embeddings = nn.ModuleList(
            [nn.Embedding(buckets, emb_dim) for _ in range(max(n_fields, 1))]
        )
        self.relation_weights = nn.Parameter(torch.ones(max(n_fields, 1)))
        self.relation_bias = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm1d(len(AGG_NAMES))
        self.projection = nn.Sequential(
            nn.Linear(len(AGG_NAMES), 32), nn.ReLU(), nn.Linear(32, out_dim), nn.Tanh()
        )
        self.head = nn.Linear(len(AGG_NAMES) + out_dim, 1)

    def soft_weights(self, cur: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid relation gates to current [B,F] and historical [B,K,F] fields."""
        if self.mode == "hard" or self.n_fields == 0:
            return torch.ones(hist.shape[0], hist.shape[1], device=hist.device)
        sims = []
        for f in range(self.n_fields):
            emb = self.embeddings[f]
            cur_vec = emb(cur[:, f]).unsqueeze(1)
            hist_vec = emb(hist[:, :, f])
            sims.append(
                torch.nn.functional.cosine_similarity(
                    hist_vec, cur_vec, dim=-1, eps=1e-08
                )
            )
        stacked = torch.stack(sims, dim=-1)
        return torch.sigmoid(stacked @ self.relation_weights + self.relation_bias)

    @staticmethod
    def aggregate(
        m_tilde: torch.Tensor,
        mask: torch.Tensor,
        z_amount: torch.Tensor,
        z_gap: torch.Tensor,
        z_amount_cur: torch.Tensor,
    ) -> torch.Tensor:
        k = mask.shape[1]
        denom = m_tilde.sum(dim=1).clamp_min(1e-06)
        wmean_amt = (m_tilde * z_amount).sum(dim=1) / denom
        wdev_amt = (m_tilde * (z_amount - z_amount_cur.unsqueeze(1)).abs()).sum(
            dim=1
        ) / denom
        wmax_amt = (m_tilde * z_amount).max(dim=1).values
        wmean_gap = (m_tilde * z_gap).sum(dim=1) / denom
        wmax_gap = (m_tilde * z_gap).max(dim=1).values
        eff_count = m_tilde.sum(dim=1) / k
        admitted = mask.sum(dim=1) / k
        big = torch.full_like(z_gap, 1000000000.0)
        recency = torch.where(mask > 0, z_gap, big).min(dim=1).values
        recency = torch.where(mask.sum(dim=1) > 0, recency, torch.zeros_like(recency))
        raw = torch.stack(
            [
                wmean_amt,
                wdev_amt,
                wmax_amt,
                wmean_gap,
                wmax_gap,
                eff_count,
                admitted,
                recency,
            ],
            dim=1,
        )
        return torch.where(
            mask.sum(dim=1, keepdim=True) > 0, raw, torch.zeros_like(raw)
        )

    def forward(
        self,
        cur: torch.Tensor,
        hist: torch.Tensor,
        mask: torch.Tensor,
        z_amount: torch.Tensor,
        z_gap: torch.Tensor,
        z_amount_cur: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        soft = self.soft_weights(cur, hist)
        m_tilde = soft * mask
        raw = self.aggregate(m_tilde, mask, z_amount, z_gap, z_amount_cur)
        normed = self.norm(raw)
        projected = self.projection(normed)
        features = torch.cat([normed, projected], dim=1)
        return (features, self.head(features).squeeze(-1))


def build_context(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    labels: np.ndarray,
    gamma: SemanticSpec,
    cfg: Dict[str, Any],
    seed: int = 2026,
    auditor=None,
) -> ContextArtifacts:
    if not bool(cfg.get("enabled", True)):
        return ContextArtifacts(
            np.zeros((len(frame), 0), dtype=np.float32),
            [],
            False,
            str(cfg.get("disabled_reason", "Disabled by configuration")),
            str(cfg.get("mode", "soft")),
            "none",
        )
    k = int(cfg["top_k"])
    mode = str(cfg.get("mode", "soft"))
    control = str(cfg.get("negative_control", "none"))
    entity_code = pd.factorize(frame["entity"].astype(str))[0].astype(np.int64)
    ts = frame["ts"].to_numpy(dtype=np.float64)
    if control == "random_history":
        hist = build_random_history_index(ts, k, seed)
    else:
        hist = build_history_index(entity_code, ts, k)
    admission = HistoryAdmission(cfg)
    mask_bool, counts = admission.evaluate(frame, hist)
    guard_checked = TemporalAvailabilityGuard.assert_history_window(
        ts, ts[np.clip(hist, 0, None)], mask_bool, float(cfg["window_seconds"])
    )
    if auditor is not None:
        auditor.record(
            "History admission: 0 < lag <= W",
            guard_checked,
            True,
            f"Admitted {counts['admitted']} / candidate slots {counts['slots_total']}",
        )
    safe = np.clip(hist, 0, None)
    amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy(dtype=np.float64)
    amount = np.nan_to_num(amount, nan=0.0)
    z_amount_all = np.log1p(np.abs(amount)).astype(np.float32)
    z_amount = z_amount_all[safe] * mask_bool
    lag = np.clip(ts[:, None] - ts[safe], 0.0, None)
    z_gap = (np.log1p(lag) * mask_bool).astype(np.float32)
    relation_fields = [
        f for f in cfg.get("soft_relation_fields", []) if f in frame.columns
    ]
    buckets = int(cfg.get("hash_buckets", 512))
    if relation_fields:
        cur_codes = np.column_stack(
            [_hash_codes(frame[f].to_numpy(), f, buckets) for f in relation_fields]
        )
        hist_codes = cur_codes[safe]
    else:
        cur_codes = np.zeros((len(frame), 0), dtype=np.int32)
        hist_codes = np.zeros((len(frame), k, 0), dtype=np.int32)
    torch.manual_seed(seed)
    model = NAGContextModel(
        len(relation_fields),
        buckets,
        int(cfg.get("embedding_dim", 8)),
        int(cfg.get("output_dim", 8)),
        mode,
    )
    tensors = dict(
        cur=torch.from_numpy(cur_codes.astype(np.int64)),
        hist=torch.from_numpy(hist_codes.astype(np.int64)),
        mask=torch.from_numpy(mask_bool.astype(np.float32)),
        z_amount=torch.from_numpy(z_amount.astype(np.float32)),
        z_gap=torch.from_numpy(z_gap),
        z_amount_cur=torch.from_numpy(z_amount_all),
    )
    training_log = _fit(model, tensors, train_idx, labels, cfg, seed)
    model.eval()
    outputs: List[np.ndarray] = []
    batch = int(cfg.get("batch_size", 4096))
    with torch.no_grad():
        for start in range(0, len(frame), batch):
            stop = min(start + batch, len(frame))
            sl = slice(start, stop)
            features, _ = model(
                tensors["cur"][sl],
                tensors["hist"][sl],
                tensors["mask"][sl],
                tensors["z_amount"][sl],
                tensors["z_gap"][sl],
                tensors["z_amount_cur"][sl],
            )
            outputs.append(features.numpy())
    matrix = np.vstack(outputs).astype(np.float32)
    names = list(AGG_NAMES) + [
        f"ctx_proj_{i}" for i in range(int(cfg.get("output_dim", 8)))
    ]
    reason = f"mode={mode}, K={k}, W={cfg['window_seconds']}s, relation fields={relation_fields or 'none'}"
    if control != "none":
        reason += f"; negative control={control}"
    return ContextArtifacts(
        matrix,
        names,
        True,
        reason,
        mode,
        control,
        admission=counts,
        training=training_log,
        n_window_checked=guard_checked,
    )


def _fit(
    model: NAGContextModel,
    tensors: Dict[str, torch.Tensor],
    train_idx: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Learn relation parameters on training data only."""
    epochs = int(cfg.get("epochs", 3))
    batch = int(cfg.get("batch_size", 4096))
    y = torch.from_numpy(labels.astype(np.float32))
    n_pos = float(max(1, int(labels[train_idx].sum())))
    n_neg = float(max(1, int((labels[train_idx] == 0).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 0.005)), weight_decay=1e-05
    )
    rng = np.random.default_rng(seed)
    log: List[Dict[str, float]] = []
    idx = np.asarray(train_idx)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(idx.size)
        total, seen = (0.0, 0)
        for start in range(0, idx.size, batch):
            rows = torch.from_numpy(idx[perm[start : start + batch]].astype(np.int64))
            if rows.numel() < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(
                tensors["cur"][rows],
                tensors["hist"][rows],
                tensors["mask"][rows],
                tensors["z_amount"][rows],
                tensors["z_gap"][rows],
                tensors["z_amount_cur"][rows],
            )
            loss = loss_fn(logits, y[rows])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * rows.numel()
            seen += int(rows.numel())
        log.append({"epoch": epoch, "mean_loss": total / max(seen, 1)})
    return {
        "epochs": log,
        "relation_weights": [float(x) for x in model.relation_weights.detach().numpy()],
        "relation_bias": float(model.relation_bias.detach().numpy()[0]),
        "n_train_rows": int(idx.size),
    }


GRAPH_FEATURE_NAMES = (
    "g_zeta",
    "g_concentration",
    "g_max_path_len",
    "g_subgraph_edges",
    "g_signal_age_buckets",
)


class CountMinSketch:
    """Fixed-memory counts with nonnegative updates and one-sided overestimation."""

    def __init__(self, width: int = 4096, depth: int = 5) -> None:
        if width <= 0 or depth <= 0:
            raise ValueError("CMS width and depth must be positive")
        self.width, self.depth = (width, depth)
        self.table = np.zeros((depth, width), dtype=np.float64)
        self._rows_cache: Dict[int, np.ndarray] = {}
        self._depth_index = np.arange(depth)

    def _rows(self, key: int) -> np.ndarray:
        cached = self._rows_cache.get(key)
        if cached is None:
            cached = np.array(
                [
                    int.from_bytes(
                        hashlib.blake2b(f"{r}::{key}".encode(), digest_size=8).digest(),
                        "little",
                    )
                    % self.width
                    for r in range(self.depth)
                ],
                dtype=np.int64,
            )
            self._rows_cache[key] = cached
        return cached

    def add(self, key: int, count: float = 1.0) -> None:
        if count < 0:
            raise ValueError("CMS does not support negative updates")
        self.table[self._depth_index, self._rows(key)] += count

    def estimate(self, key: int) -> float:
        return float(self.table[self._depth_index, self._rows(key)].min())


def midas_statistic(current: np.ndarray, cumulative: np.ndarray, t: int) -> np.ndarray:
    if t <= 1:
        return np.zeros_like(current, dtype=np.float64)
    expected = cumulative / t
    denom = np.maximum(cumulative * (t - 1), 1e-12)
    return (current - expected) ** 2 * t**2 / denom


def chi_square_threshold(epsilon_t: float) -> float:
    """Return the chi-squared burst-screening threshold."""
    if not 0.0 < epsilon_t < 1.0:
        raise ValueError("epsilon_t must be in (0,1)")
    return float(chi2.isf(epsilon_t / 2.0, 1))


def vectorize_paths(
    paths: Sequence[Sequence[Tuple[int, int, int]]],
    dim: int,
    node_names: Dict[int, str],
    relation_names: Dict[int, str],
) -> np.ndarray:
    out = np.zeros((len(paths), dim), dtype=np.float64)
    for row, path in enumerate(paths):
        for src, rel, dst in path:
            for token in (
                f"src:{node_names[src]}",
                f"rel:{relation_names[rel]}",
                f"dst:{node_names[dst]}",
            ):
                raw = int.from_bytes(
                    hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
                    "little",
                )
                out[row, raw % dim] += 1.0 if raw >> 17 & 1 == 0 else -1.0
        norm = np.linalg.norm(out[row])
        if norm > 0:
            out[row] /= norm
    return out


def path_attention(vectors: np.ndarray) -> np.ndarray:
    """Compute path similarities using scaled identity projections, not learned attention."""
    if vectors.shape[0] == 0:
        return np.zeros((0, 0))
    logits = vectors @ vectors.T / math.sqrt(max(1, vectors.shape[1]))
    spread = float(logits.std())
    if spread > 1e-09:
        logits = (logits - float(logits.mean())) / spread
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def attention_concentration(attention: np.ndarray) -> float:
    """Measure concentration as normalized attention-entropy reduction."""
    k = attention.shape[0]
    if k <= 1:
        return 0.0
    entropy = -float(np.sum(attention * np.log(np.clip(attention, 1e-12, 1.0))) / k)
    return float(np.clip(1.0 - entropy / math.log(k), 0.0, 1.0))


def path_overlap_concentration(
    paths: Sequence[Sequence[Tuple[int, int, int]]]
) -> float:
    """Measure edge reuse as one minus unique edges divided by total path length."""
    total = sum((len(path) for path in paths))
    if total <= 1:
        return 0.0
    distinct = len({edge for path in paths for edge in path})
    return float(np.clip(1.0 - distinct / total, 0.0, 1.0))


def edge_usage_ranking(paths: Sequence[Sequence[Tuple[int, int, int]]]) -> np.ndarray:
    """Rank paths by the reuse counts of their edges."""
    counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for path in paths:
        for edge in set(path):
            counts[edge] += 1
    return np.array(
        [float(np.mean([counts[e] for e in path])) if path else 0.0 for path in paths]
    )


def pattern_signature(
    nodes: Sequence[str],
    relations: Sequence[Tuple[str, str, str]],
    times: Sequence[float],
    attributes: Sequence[str],
    gap_bin_seconds: float = 3600.0,
) -> str:
    """Build an entity-ID-independent structural and temporal signature."""
    in_deg: Dict[str, int] = defaultdict(int)
    out_deg: Dict[str, int] = defaultdict(int)
    rels: List[str] = []
    for src, rel, dst in relations:
        out_deg[src] += 1
        in_deg[dst] += 1
        rels.append(rel)
    degree_profile = sorted(((out_deg[n], in_deg[n]) for n in nodes))
    ordered = sorted(times)
    gaps = tuple(
        (int((b - a) // gap_bin_seconds) for a, b in zip(ordered, ordered[1:]))
    )
    payload = repr(
        (
            len(nodes),
            degree_profile,
            tuple(sorted(rels)),
            gaps,
            tuple(sorted(attributes)),
        )
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


class _Adjacency:
    """Append edges incrementally while maintaining an as-of graph view."""

    def __init__(self) -> None:
        self.out: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
        self.n_edges = 0
        self.max_ts = -np.inf

    def add(self, src: int, dst: int, rel: int, ts: float) -> None:
        self.out[src].append((dst, rel, ts))
        self.n_edges += 1
        if ts > self.max_ts:
            self.max_ts = ts


def _sample_paths(
    adj: _Adjacency,
    seeds: Tuple[int, int],
    k: int,
    max_hops: int,
    rng: np.random.Generator,
) -> List[List[Tuple[int, int, int]]]:
    """Sample typed paths from both seed endpoints within the configured hop limit."""
    paths: List[List[Tuple[int, int, int]]] = []
    for i in range(k):
        node = seeds[i % 2]
        hops = int(rng.integers(1, max_hops + 1))
        path: List[Tuple[int, int, int]] = []
        used: set = set()
        for _ in range(hops):
            candidates = adj.out.get(node)
            if not candidates:
                break
            dst, rel, edge_ts = candidates[int(rng.integers(0, len(candidates)))]
            key = (node, dst, rel, edge_ts)
            if key in used:
                break
            used.add(key)
            path.append((node, rel, dst))
            node = dst
        if path:
            paths.append(path)
    return paths


@dataclass(frozen=True)
class GraphConfirmation:
    """Store discovered evidence before selecting emission thresholds and weights."""

    bucket: int
    targets: Tuple[int, ...]
    burst: float
    rho: float
    max_path_len: float
    n_subgraph_edges: float
    novelty: str = "UNKNOWN"

    def values(self, zeta_mode: str) -> np.ndarray:
        """Build graph-evidence values using the selected intensity mode."""
        zeta = self.burst * self.rho if zeta_mode == "product" else self.burst
        return np.array(
            [zeta, self.rho, self.max_path_len, self.n_subgraph_edges], dtype=np.float32
        )


@dataclass
class GraphEmissionContext:
    """Inputs reused when replaying graph-evidence emission."""

    n_rows: int
    bucket_rows: List[Tuple[int, np.ndarray]]
    endpoint_nodes: List[np.ndarray]
    horizon: int


def emit_graph_features(
    ctx: GraphEmissionContext,
    confirmations: Sequence[GraphConfirmation],
    threshold: float,
    zeta_mode: str = "product",
) -> Tuple[np.ndarray, int]:
    """Emit past-bucket evidence using separate concentration gates and intensity weights."""
    features = np.zeros((ctx.n_rows, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
    by_bucket: Dict[int, List[Tuple[np.ndarray, GraphConfirmation]]] = defaultdict(list)
    n_passed = 0
    for item in confirmations:
        if item.rho >= threshold:
            by_bucket[item.bucket].append((item.values(zeta_mode), item))
            n_passed += 1
    node_state: Dict[int, Tuple[np.ndarray, int]] = {}
    for b, rows in ctx.bucket_rows:
        if node_state:
            expired = [nd for nd, (_, at) in node_state.items() if b - at > ctx.horizon]
            for nd in expired:
                del node_state[nd]
        if node_state and rows.size:
            for endpoint in ctx.endpoint_nodes:
                nodes = endpoint[rows]
                for j in range(rows.size):
                    entry = node_state.get(int(nodes[j]))
                    if entry is None:
                        continue
                    values, confirmed_at = entry
                    row = int(rows[j])
                    if values[0] > features[row, 0]:
                        features[row] = np.append(values, float(b - confirmed_at))
        for values, item in by_bucket.get(b, ()):
            for node in item.targets:
                previous = node_state.get(node)
                if previous is None or values[0] >= previous[0][0]:
                    node_state[node] = (values, b)
    return (features, n_passed)


@dataclass
class GraphArtifacts:
    matrix: np.ndarray
    feature_names: List[str]
    enabled: bool
    reason: str
    stats: Dict[str, Any] = field(default_factory=dict)
    patterns: List[PatternEntry] = field(default_factory=list)
    confirmations: List[GraphConfirmation] = field(default_factory=list)
    emission: Optional[GraphEmissionContext] = None
    rho_samples_train: List[float] = field(default_factory=list)
    _describe: Optional[Any] = None

    def resolve_threshold(
        self, setting: Any, quantile: float, minimum_seeds: int, fallback: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Resolve the concentration threshold and record its training calibration."""
        if not (isinstance(setting, str) and setting.lower() == "auto"):
            return (float(setting), {"mode": "fixed", "threshold": float(setting)})
        samples = self.rho_samples_train
        if len(samples) >= minimum_seeds:
            threshold, mode = (float(np.quantile(samples, quantile)), "auto")
        else:
            threshold, mode = (float(fallback), "auto_unavailable")
        return (
            threshold,
            {
                "mode": mode,
                "quantile": quantile,
                "n_train_seeds": len(samples),
                "min_required_seeds": minimum_seeds,
                "threshold": threshold,
                "train_rho_quantiles": {
                    str(q): float(np.quantile(samples, q)) if samples else None
                    for q in (0.1, 0.25, 0.5, 0.75, 0.9)
                },
            },
        )

    def with_threshold(
        self, threshold: float, calibration: Dict[str, Any], zeta_mode: str = "product"
    ) -> "GraphArtifacts":
        """Replay emission at a new threshold without repeating graph discovery."""
        if not self.enabled or self.emission is None:
            return self
        matrix, n_passed = emit_graph_features(
            self.emission, self.confirmations, threshold, zeta_mode
        )
        stats = dict(self.stats)
        stats["n_structure_confirmed"] = n_passed
        stats["n_time_only"] = stats["n_expanded"] - stats["n_no_path"] - n_passed
        stats["n_rows_with_signal"] = int((matrix[:, 0] > 0).sum())
        stats["rho_min_used"] = threshold
        stats["zeta_mode_used"] = zeta_mode
        stats["rho_calibration"] = calibration
        novelty: Dict[str, int] = defaultdict(int)
        for item in self.confirmations:
            if item.rho >= threshold:
                novelty[item.novelty] += 1
        kept: List[PatternEntry] = []
        for pattern in self.patterns:
            if pattern.concentration >= threshold:
                kept.append(pattern)
        stats["novelty"] = dict(novelty)
        stats["n_patterns_retained_for_audit"] = len(kept)
        reason = (
            self._describe(threshold, calibration) if self._describe else self.reason
        )
        return GraphArtifacts(
            matrix,
            list(self.feature_names),
            True,
            reason,
            stats,
            kept,
            self.confirmations,
            self.emission,
            self.rho_samples_train,
            self._describe,
        )


def _pattern_attributes(
    frame: pd.DataFrame, rows: np.ndarray, gamma: SemanticSpec
) -> Dict[str, Any]:
    """Extract pattern attributes allowed by the shared field schema."""
    attrs: Dict[str, Any] = {"n_transactions": int(rows.size)}
    for name in ("amount", "cat1", "cat2", "cat3"):
        if name not in frame.columns or name not in gamma.fields:
            continue
        spec = gamma.field(name)
        if spec.posthoc or spec.availability_lag_seconds > 0:
            continue
        values = frame[name].to_numpy()[rows]
        if name == "amount":
            numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = numeric[np.isfinite(numeric)]
            attrs["amount_sum"] = float(finite.sum()) if finite.size else 0.0
            attrs["amount_max"] = float(finite.max()) if finite.size else 0.0
            attrs["amount_unit"] = spec.unit
        else:
            attrs[name] = sorted({spec.decode(v) for v in values})[:5]
    return attrs


def build_graph(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    gamma: SemanticSpec,
    cfg: Dict[str, Any],
    seed: int = 2026,
    auditor=None,
) -> GraphArtifacts:
    """Discover candidate patterns and emit features at the configured threshold."""
    if not bool(cfg.get("enabled", True)):
        return GraphArtifacts(
            np.zeros((len(frame), 0), dtype=np.float32),
            [],
            False,
            str(cfg.get("disabled_reason", "Disabled by configuration")),
        )
    n = len(frame)
    ts = frame["ts"].to_numpy(dtype=np.float64)
    bucket_seconds = float(cfg["bucket_seconds"])
    bucket = np.floor((ts - ts.min()) / bucket_seconds).astype(np.int64)
    streams = list(cfg.get("streams") or [])
    if not streams:
        raise ValueError("graph.streams must contain at least one edge stream")
    node_id: Dict[str, int] = {}
    node_names: Dict[int, str] = {}
    relation_names: Dict[int, str] = {}

    def nid(namespace: str, value: str) -> int:
        key = f"{namespace}:{value}"
        found = node_id.get(key)
        if found is None:
            found = len(node_id)
            node_id[key] = found
            node_names[found] = key
        return found

    edge_src: List[np.ndarray] = []
    edge_dst: List[np.ndarray] = []
    edge_rel: List[int] = []
    for s_index, spec in enumerate(streams):
        src_ns, dst_ns = (
            str(spec.get("source_ns", "N")),
            str(spec.get("target_ns", "N")),
        )
        src_vals = frame[spec["source"]].astype(str).to_numpy()
        dst_vals = frame[spec["target"]].astype(str).to_numpy()
        rel = len(relation_names)
        relation_names[rel] = str(spec.get("relation", f"rel{s_index}"))
        edge_src.append(
            np.fromiter((nid(src_ns, v) for v in src_vals), dtype=np.int64, count=n)
        )
        edge_dst.append(
            np.fromiter((nid(dst_ns, v) for v in dst_vals), dtype=np.int64, count=n)
        )
        edge_rel.append(rel)
    n_nodes = max(len(node_id), 1)
    add_reverse = bool(cfg.get("add_reverse", True))
    reverse_rel: Dict[int, int] = {}
    if add_reverse:
        for rel in list(relation_names):
            new_rel = len(relation_names)
            relation_names[new_rel] = "rev_" + relation_names[rel]
            reverse_rel[rel] = new_rel
    trigger_streams = [
        i for i, s in enumerate(streams) if bool(s.get("trigger", i == 0))
    ]
    pair_key_row = {i: edge_src[i] * n_nodes + edge_dst[i] for i in trigger_streams}
    epsilon = float(cfg["epsilon"])
    bonferroni = bool(cfg.get("bonferroni", True))
    use_trigger = bool(cfg.get("trigger", True))
    require_prior = bool(cfg.get("require_prior_history", True))
    force_rho_one = bool(cfg.get("force_rho_one", False))
    max_seeds = int(cfg.get("max_seeds_per_bucket", 64))
    k_paths = int(cfg.get("path_samples", 16))
    max_hops = int(cfg.get("max_hops", 3))
    vector_dim = int(cfg.get("vector_dim", 32))
    counter_mode = str(cfg.get("counter", "exact"))
    max_patterns = int(cfg.get("max_patterns_exported", 2000))
    horizon = int(cfg.get("signal_horizon_buckets", 24))
    signal_scope = str(cfg.get("signal_scope", "seed"))
    rho_estimator = str(cfg.get("rho_estimator", "overlap"))
    if rho_estimator not in ("overlap", "attention"):
        raise ValueError("graph.rho_estimator must be overlap or attention")
    rho_attn_sum, rho_attn_count = (0.0, 0)
    last_train_bucket = (
        int(bucket[train_idx].max()) if train_idx.size else int(bucket.max())
    )
    rho_samples: List[float] = []
    rng = np.random.default_rng(seed)
    adjacency = _Adjacency()
    patterns: List[PatternEntry] = []
    train_replacement_cursor = max_patterns - 1
    confirmations: List[GraphConfirmation] = []
    bucket_rows: List[Tuple[int, np.ndarray]] = []
    cumulative: Dict[int, float] = defaultdict(float)
    sketch = (
        CountMinSketch(int(cfg.get("cms_width", 4096)), int(cfg.get("cms_depth", 5)))
        if counter_mode == "cms"
        else None
    )
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    train_signatures: set = set()
    stats: Dict[str, Any] = {
        "n_buckets": 0,
        "n_pair_tests": 0,
        "n_triggered": 0,
        "n_expanded": 0,
        "n_structure_confirmed": 0,
        "n_time_only": 0,
        "n_no_path": 0,
        "n_seed_cap_hits": 0,
        "n_skipped_no_prior_history": 0,
        "n_edges_in_adjacency": 0,
        "n_rows_with_signal": 0,
        "trigger_rate_head": [],
        "m_t_head": [],
        "epsilon_t_head": [],
        "novelty": defaultdict(int),
    }
    for tick, b in enumerate(np.unique(bucket), start=1):
        rows = np.flatnonzero(bucket == b)
        decision_time = float(ts[rows].max())
        stats["n_buckets"] += 1
        bucket_rows.append((int(b), rows))
        seeds: List[Dict[str, Any]] = []
        for s_index in trigger_streams:
            pairs = pair_key_row[s_index][rows]
            uniq, counts_now = np.unique(pairs, return_counts=True)
            m_t = int(uniq.size)
            stats["n_pair_tests"] += m_t
            eps_t = epsilon / max(m_t, 1) if bonferroni else epsilon
            threshold = chi_square_threshold(eps_t)
            if len(stats["m_t_head"]) < 50:
                stats["m_t_head"].append(m_t)
                stats["epsilon_t_head"].append(float(eps_t))
            prior = np.array(
                [
                    (
                        sketch.estimate(int(k))
                        if sketch is not None
                        else cumulative[int(k)]
                    )
                    for k in uniq
                ],
                dtype=np.float64,
            )
            current = counts_now.astype(np.float64)
            if sketch is not None:
                for key, count in zip(uniq, current):
                    sketch.add(int(key), float(count))
                total = np.array(
                    [sketch.estimate(int(k)) for k in uniq], dtype=np.float64
                )
            else:
                for key, count in zip(uniq, current):
                    cumulative[int(key)] += float(count)
                total = prior + current
            scores = midas_statistic(current, total, tick)
            eligible = np.ones(m_t, dtype=bool)
            if require_prior:
                eligible = prior > 0
                stats["n_skipped_no_prior_history"] += int((~eligible).sum())
            triggered = (
                (scores > threshold) & eligible if use_trigger else eligible.copy()
            )
            trigger_idx = np.flatnonzero(triggered)
            stats["n_triggered"] += int(trigger_idx.size)
            if len(stats["trigger_rate_head"]) < 50:
                stats["trigger_rate_head"].append(float(trigger_idx.size / max(m_t, 1)))
            if trigger_idx.size > max_seeds:
                stats["n_seed_cap_hits"] += 1
                trigger_idx = trigger_idx[
                    np.argsort(scores[trigger_idx])[::-1][:max_seeds]
                ]
            for i in trigger_idx:
                key = int(uniq[i])
                seeds.append(
                    {
                        "stream": s_index,
                        "pair_key": key,
                        "pair": (key // n_nodes, key % n_nodes),
                        "score": float(scores[i]),
                        "eps_t": float(eps_t),
                        "m_t": m_t,
                        "rows": rows[pairs == key],
                    }
                )
        for s_index in range(len(streams)):
            src, dst = (edge_src[s_index][rows], edge_dst[s_index][rows])
            rel = edge_rel[s_index]
            for j in range(rows.size):
                edge_ts = float(ts[rows[j]])
                adjacency.add(int(src[j]), int(dst[j]), rel, edge_ts)
                if add_reverse:
                    adjacency.add(int(dst[j]), int(src[j]), reverse_rel[rel], edge_ts)
        if adjacency.max_ts > decision_time:
            raise ContractViolation(
                f"Future edge in adjacency: max(edge_time)={adjacency.max_ts} > t={decision_time}"
            )
        for seed_info in seeds:
            stats["n_expanded"] += 1
            paths = _sample_paths(adjacency, seed_info["pair"], k_paths, max_hops, rng)
            if not paths:
                stats["n_no_path"] += 1
                continue
            attention = path_attention(
                vectorize_paths(paths, vector_dim, node_names, relation_names)
            )
            rho_attn = attention_concentration(attention)
            rho_overlap = path_overlap_concentration(paths)
            rho_attn_sum += rho_attn
            rho_attn_count += 1
            rho = (
                1.0
                if force_rho_one
                else rho_attn if rho_estimator == "attention" else rho_overlap
            )
            if b <= last_train_bucket:
                rho_samples.append(float(rho))
            zeta = seed_info["score"] * rho
            contribution = edge_usage_ranking(paths)
            keep = max(1, math.ceil(len(paths) * 0.25))
            chosen = np.argsort(contribution)[-keep:][::-1]
            edges: Dict[Tuple[int, int, int], Tuple[str, str, str]] = {}
            for p in chosen:
                for src, rel, dst in paths[int(p)]:
                    edges[src, rel, dst] = (
                        node_names[src],
                        relation_names[rel],
                        node_names[dst],
                    )
            subgraph_nodes = sorted(
                {x for triple in edges.values() for x in (triple[0], triple[2])}
            )
            relations = tuple(sorted(edges.values()))
            times = tuple((float(x) for x in ts[seed_info["rows"]]))
            attrs = _pattern_attributes(frame, seed_info["rows"], gamma)
            signature = pattern_signature(
                subgraph_nodes,
                relations,
                times,
                tuple((k for k in attrs if k != "n_transactions")),
            )
            if bool(train_mask[seed_info["rows"]].all()):
                train_signatures.add(signature)
                novelty = "TRAIN_PERIOD"
            else:
                novelty = (
                    "MATCHES_TRAIN"
                    if signature in train_signatures
                    else "UNSEEN_CANDIDATE"
                )
            stats["novelty"][novelty] += 1
            targets = set(seed_info["pair"])
            if signal_scope == "subgraph":
                targets |= {n for src, _, dst in edges for n in (src, dst)}
            confirmations.append(
                GraphConfirmation(
                    int(b),
                    tuple(sorted(targets)),
                    float(seed_info["score"]),
                    float(rho),
                    float(max((len(p) for p in paths))),
                    float(len(edges)),
                    novelty,
                )
            )
            entry = PatternEntry(
                pattern_id=hashlib.blake2b(
                    repr((signature, seed_info["pair_key"], int(b))).encode(),
                    digest_size=12,
                ).hexdigest(),
                S=tuple(subgraph_nodes),
                R=relations,
                T=times,
                A=attrs,
                seed_pair=(
                    node_names[seed_info["pair"][0]],
                    node_names[seed_info["pair"][1]],
                ),
                time_bucket=int(b),
                burst_score=seed_info["score"],
                concentration=rho,
                zeta=zeta,
                epsilon_t=seed_info["eps_t"],
                m_t=seed_info["m_t"],
                novelty=novelty,
                signature=signature,
            )
            if len(patterns) < max_patterns:
                patterns.append(entry)
            elif novelty != "TRAIN_PERIOD" and train_replacement_cursor >= 0:
                while (
                    train_replacement_cursor >= 0
                    and patterns[train_replacement_cursor].novelty != "TRAIN_PERIOD"
                ):
                    train_replacement_cursor -= 1
                if train_replacement_cursor >= 0:
                    patterns[train_replacement_cursor] = entry
                    train_replacement_cursor -= 1
    stats["n_edges_in_adjacency"] = adjacency.n_edges
    stats["rho_estimator"] = rho_estimator
    stats["mean_rho_attention"] = rho_attn_sum / max(rho_attn_count, 1)
    stats["mean_trigger_rate"] = (
        float(np.mean(stats["trigger_rate_head"]))
        if stats["trigger_rate_head"]
        else 0.0
    )
    stats["mean_m_t"] = float(np.mean(stats["m_t_head"])) if stats["m_t_head"] else 0.0
    stats["expansion_call_ratio"] = stats["n_expanded"] / max(stats["n_pair_tests"], 1)
    stats["novelty"] = dict(stats["novelty"])

    def describe(threshold: float, calibration: Dict[str, Any]) -> str:
        return f"streams={[s.get('relation') for s in streams]}, bucket={bucket_seconds:g}s, ε={epsilon}, trigger={('on' if use_trigger else 'off(expand all entity pairs)')}, ϱ_{rho_estimator}={('forced=1.0' if force_rho_one else f'≥{threshold:.4f}')}({calibration['mode']}), evidence horizon={horizon} buckets"

    emission = GraphEmissionContext(
        n_rows=n,
        bucket_rows=bucket_rows,
        endpoint_nodes=[
            arr
            for s_index in range(len(streams))
            for arr in (edge_src[s_index], edge_dst[s_index])
        ],
        horizon=horizon,
    )
    base = GraphArtifacts(
        np.zeros((n, len(GRAPH_FEATURE_NAMES)), dtype=np.float32),
        list(GRAPH_FEATURE_NAMES),
        True,
        "",
        stats,
        patterns,
        confirmations,
        emission,
        rho_samples,
        describe,
    )
    quantile_setting = cfg.get("rho_quantile", 0.75)
    if isinstance(quantile_setting, str) and quantile_setting.lower() == "auto":
        grid = [float(q) for q in cfg.get("rho_quantile_grid", [0.75])] or [0.75]
        quantile_setting = grid[len(grid) // 2]
    threshold, calibration = base.resolve_threshold(
        cfg.get("rho_min", "auto"),
        float(quantile_setting),
        int(cfg.get("rho_min_calibration_seeds", 20)),
        float(cfg.get("rho_fallback", 0.2)),
    )
    if force_rho_one:
        threshold, calibration = (
            -np.inf,
            {"mode": "forced_rho_one", "threshold": -np.inf},
        )
    zeta_setting = str(cfg.get("zeta_mode", "product"))
    if zeta_setting.lower() == "auto":
        zeta_setting = "product"
    result = base.with_threshold(threshold, calibration, zeta_setting)
    if auditor is not None:
        auditor.record(
            "Graph queries use only available edges",
            adjacency.n_edges,
            True,
            f"{stats['n_buckets']} buckets appended incrementally; evidence is available after confirmation in buckets 1..{horizon}",
        )
        auditor.record(
            "Concentration thresholds use training data only",
            int(calibration.get("n_train_seeds", 0)),
            calibration["mode"] != "auto_unavailable",
            f"mode={calibration['mode']}, threshold={threshold:.4f}"
            + (
                " (insufficient training seeds; using fallback threshold)"
                if calibration["mode"] == "auto_unavailable"
                else ""
            ),
        )
        auditor.record(
            "Discovery filtering preserves a subset of candidates",
            stats["n_expanded"],
            True,
            f"Triggered {stats['n_triggered']} -> expanded {stats['n_expanded']}; confirmation count depends on the selected concentration threshold",
        )
    return result


@dataclass
class DetectorArtifacts:
    best_iteration: int
    best_score: float
    n_features: int
    params: Dict[str, Any]
    global_importance: List[Tuple[str, float]] = field(default_factory=list)


class FusedDetector:
    """XGBoost fusion scorer with early stopping on the tuning segment."""

    def __init__(
        self, cfg: Dict[str, Any], feature_names: Sequence[str], seed: int = 2026
    ) -> None:
        self.cfg = cfg
        self.feature_names = list(feature_names)
        self.seed = seed
        self.model: Optional[xgb.XGBClassifier] = None
        self.artifacts: Optional[DetectorArtifacts] = None

    def _params(self, y_train: np.ndarray) -> Dict[str, Any]:
        spw = self.cfg.get("scale_pos_weight", "auto")
        if spw == "auto":
            n_pos = max(1, int(y_train.sum()))
            spw = float((y_train.size - n_pos) / n_pos)
        return dict(
            n_estimators=int(self.cfg.get("n_estimators", 600)),
            max_depth=int(self.cfg.get("max_depth", 6)),
            learning_rate=float(self.cfg.get("learning_rate", 0.05)),
            subsample=float(self.cfg.get("subsample", 0.8)),
            colsample_bytree=float(self.cfg.get("colsample_bytree", 0.8)),
            min_child_weight=float(self.cfg.get("min_child_weight", 5)),
            reg_lambda=float(self.cfg.get("reg_lambda", 1.0)),
            scale_pos_weight=float(spw),
            tree_method="hist",
            eval_metric="aucpr",
            objective="binary:logistic",
            random_state=self.seed,
            n_jobs=int(self.cfg.get("n_jobs", -1)),
            early_stopping_rounds=int(self.cfg.get("early_stopping_rounds", 50)),
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_tune: np.ndarray,
        y_tune: np.ndarray,
    ) -> DetectorArtifacts:
        params = self._params(y_train)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_tune, y_tune)], verbose=False)
        booster = self.model.get_booster()
        booster.feature_names = self.feature_names
        gain = booster.get_score(importance_type="total_gain")
        importance = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:25]
        self.artifacts = DetectorArtifacts(
            best_iteration=int(
                getattr(self.model, "best_iteration", params["n_estimators"] - 1)
            ),
            best_score=float(getattr(self.model, "best_score", float("nan"))),
            n_features=X_train.shape[1],
            params=params,
            global_importance=[(k, float(v)) for k, v in importance],
        )
        return self.artifacts

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit before predict")
        return self.model.predict_proba(X)[:, 1].astype(np.float64)

    def shap_contributions(self, X: np.ndarray) -> np.ndarray:
        """Return TreeSHAP feature contributions, excluding the base-value column."""
        if self.model is None:
            raise RuntimeError("Call fit before attribution")
        booster = self.model.get_booster()
        matrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return booster.predict(matrix, pred_contribs=True)[:, :-1]

    def explain_top_k(self, X: np.ndarray, k: int = 3) -> List[List[Dict[str, float]]]:
        contribs = self.shap_contributions(X)
        out: List[List[Dict[str, float]]] = []
        for row in contribs:
            order = np.argsort(np.abs(row))[::-1][:k]
            out.append(
                [
                    {
                        "feature": self.feature_names[int(i)],
                        "impact": float(row[int(i)]),
                    }
                    for i in order
                ]
            )
        return out


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
