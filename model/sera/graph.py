from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2
from .contracts import ContractViolation, PatternEntry, SemanticSpec

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
