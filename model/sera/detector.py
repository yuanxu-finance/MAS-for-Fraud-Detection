from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import xgboost as xgb


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
