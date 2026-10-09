from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from .contracts import SemanticSpec, TemporalAvailabilityGuard

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
