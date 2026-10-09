"""SERA evidence construction, fusion scoring, and calibrated evaluation."""

from __future__ import annotations
import copy
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .sera.config import (
    DATASETS,
    config_digest,
    get_path,
    load_model_config as load_config,
)
from .sera.constraint import CalibrationInfeasible, CapacityGate, make_calibrator
from .sera.context import ContextArtifacts, build_context
from .sera.contracts import ContractAuditor, ScorerLifecycle, SemanticSpec
from .sera.datasets import Workspace
from .sera.detector import FusedDetector
from .sera.graph import GraphArtifacts, build_graph
from .sera.metrics import average_precision, evaluate, tpr_at_fpr
from .sera.semantic import build_semantic


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
