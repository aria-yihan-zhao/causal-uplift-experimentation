"""Uplift learners, locked data splits, and treatment-aware evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import FEATURE_COLUMNS, validate_data_contract


def locked_split(
    frame: pd.DataFrame,
    outcome: str = "visit",
    seed: int = 20261112,
) -> dict[str, np.ndarray]:
    """Create deterministic 60/20/20 splits stratified by arm and outcome."""
    validate_data_contract(frame)
    indices = np.arange(len(frame))
    strata = frame["treatment"].to_numpy(dtype=int) * 2 + frame[outcome].to_numpy(dtype=int)
    train_validation, test = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=strata,
    )
    train_validation_strata = strata[train_validation]
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed + 1,
        stratify=train_validation_strata,
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


def split_manifest(
    frame: pd.DataFrame,
    splits: dict[str, np.ndarray],
    outcome: str,
    seed: int,
) -> dict[str, Any]:
    """Summarize a locked split without exposing row-level indices."""
    payload: dict[str, Any] = {
        "seed": seed,
        "outcome": outcome,
        "stratification": ["treatment", outcome],
        "split_fractions": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "splits": {},
    }
    for name, indices in splits.items():
        subset = frame.iloc[indices]
        payload["splits"][name] = {
            "rows": int(len(subset)),
            "treatment_share": float(subset["treatment"].mean()),
            "outcome_rate": float(subset[outcome].mean()),
        }
    return payload


def _classifier(seed: int, max_iter: int = 100) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=seed,
    )


def _regressor(seed: int, max_iter: int = 100) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=max_iter,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=seed,
    )


@dataclass
class SLearner:
    """Single outcome model using treatment as a feature."""

    seed: int = 20261112
    max_iter: int = 100

    def fit(self, features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "SLearner":
        self.model_ = _classifier(self.seed, self.max_iter)
        augmented = np.column_stack([features, treatment])
        self.model_.fit(augmented, outcome)
        return self

    def predict_uplift(self, features: np.ndarray) -> np.ndarray:
        treated = np.column_stack([features, np.ones(len(features))])
        control = np.column_stack([features, np.zeros(len(features))])
        return self.model_.predict_proba(treated)[:, 1] - self.model_.predict_proba(control)[:, 1]


@dataclass
class TLearner:
    """Separate treated and control outcome models."""

    seed: int = 20261112
    max_iter: int = 100

    def fit(self, features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "TLearner":
        self.control_model_ = _classifier(self.seed, self.max_iter)
        self.treated_model_ = _classifier(self.seed + 1, self.max_iter)
        self.control_model_.fit(features[treatment == 0], outcome[treatment == 0])
        self.treated_model_.fit(features[treatment == 1], outcome[treatment == 1])
        return self

    def predict_uplift(self, features: np.ndarray) -> np.ndarray:
        return (
            self.treated_model_.predict_proba(features)[:, 1]
            - self.control_model_.predict_proba(features)[:, 1]
        )


@dataclass
class DRLearner:
    """Cross-fitted doubly robust learner with known randomized propensity."""

    seed: int = 20261112
    max_iter: int = 100
    folds: int = 2

    def fit(self, features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "DRLearner":
        propensity = float(treatment.mean())
        if not 0.01 < propensity < 0.99:
            raise ValueError("Treatment propensity is too close to zero or one.")

        mu0 = np.empty(len(features), dtype=float)
        mu1 = np.empty(len(features), dtype=float)
        strata = treatment.astype(int) * 2 + outcome.astype(int)
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)

        for fold, (fit_indices, holdout_indices) in enumerate(splitter.split(features, strata)):
            fit_treatment = treatment[fit_indices]
            control_model = _classifier(self.seed + 10 + fold * 2, self.max_iter)
            treated_model = _classifier(self.seed + 11 + fold * 2, self.max_iter)
            control_model.fit(
                features[fit_indices][fit_treatment == 0],
                outcome[fit_indices][fit_treatment == 0],
            )
            treated_model.fit(
                features[fit_indices][fit_treatment == 1],
                outcome[fit_indices][fit_treatment == 1],
            )
            mu0[holdout_indices] = control_model.predict_proba(features[holdout_indices])[:, 1]
            mu1[holdout_indices] = treated_model.predict_proba(features[holdout_indices])[:, 1]

        pseudo_outcome = (
            mu1
            - mu0
            + treatment * (outcome - mu1) / propensity
            - (1 - treatment) * (outcome - mu0) / (1 - propensity)
        )
        self.effect_model_ = _regressor(self.seed + 100, self.max_iter)
        self.effect_model_.fit(features, pseudo_outcome)
        self.propensity_ = propensity
        return self

    def predict_uplift(self, features: np.ndarray) -> np.ndarray:
        return self.effect_model_.predict(features)


def ipw_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensity: float,
) -> np.ndarray:
    """Return inverse-propensity transformed outcomes for randomized data."""
    if not 0 < propensity < 1:
        raise ValueError("propensity must be between zero and one.")
    return treatment * outcome / propensity - (1 - treatment) * outcome / (1 - propensity)


def uplift_curve(
    outcome: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    propensity: float | None = None,
    points: int = 101,
) -> pd.DataFrame:
    """Construct a normalized cumulative IPW uplift curve."""
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    uplift_score = np.asarray(uplift_score, dtype=float)
    if not (len(outcome) == len(treatment) == len(uplift_score)) or len(outcome) == 0:
        raise ValueError("Outcome, treatment, and score arrays must have equal nonzero length.")
    if not np.isfinite(uplift_score).all():
        raise ValueError("Uplift scores must be finite.")
    if propensity is None:
        propensity = float(treatment.mean())

    order = np.argsort(-uplift_score, kind="mergesort")
    transformed = ipw_scores(outcome[order], treatment[order], propensity)
    cumulative_gain = np.cumsum(transformed) / len(transformed)
    population_fraction = np.arange(1, len(transformed) + 1) / len(transformed)
    selected = np.unique(
        np.concatenate(
            [[0], np.linspace(0, len(transformed) - 1, min(points - 1, len(transformed)), dtype=int)]
        )
    )
    curve = pd.DataFrame(
        {
            "population_fraction": np.concatenate([[0.0], population_fraction[selected]]),
            "cumulative_incremental_outcome": np.concatenate([[0.0], cumulative_gain[selected]]),
        }
    )
    return curve.drop_duplicates("population_fraction", keep="last").reset_index(drop=True)


def uplift_metrics(
    outcome: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    propensity: float | None = None,
) -> dict[str, float]:
    """Compute normalized AUUC and Qini relative to random targeting."""
    curve = uplift_curve(outcome, treatment, uplift_score, propensity, points=len(outcome) + 1)
    x = curve["population_fraction"].to_numpy()
    gain = curve["cumulative_incremental_outcome"].to_numpy()
    auuc = float(np.trapezoid(gain, x))
    final_gain = float(gain[-1])
    random_auuc = 0.5 * final_gain
    return {
        "auuc": auuc,
        "qini": auuc - random_auuc,
        "full_population_ipw_ate": final_gain,
    }


def policy_value_table(
    outcome: np.ndarray,
    treatment: np.ndarray,
    uplift_score: np.ndarray,
    model_name: str,
    propensity: float | None = None,
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50, 1.00),
) -> pd.DataFrame:
    """Estimate incremental outcomes under top-score targeting policies."""
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    uplift_score = np.asarray(uplift_score, dtype=float)
    if propensity is None:
        propensity = float(treatment.mean())
    transformed = ipw_scores(outcome, treatment, propensity)
    order = np.argsort(-uplift_score, kind="mergesort")

    rows: list[dict[str, float | str | int]] = []
    for fraction in fractions:
        selected_count = max(1, int(round(fraction * len(outcome))))
        selected = order[:selected_count]
        value = float(transformed[selected].sum() / len(outcome))
        rows.append(
            {
                "model": model_name,
                "target_fraction": fraction,
                "targeted_customers": selected_count,
                "incremental_outcomes_per_eligible_customer": value,
                "incremental_outcomes_per_10000_eligible": value * 10_000,
                "average_incremental_outcome_among_targeted": value / fraction,
            }
        )
    return pd.DataFrame(rows)


def frame_arrays(frame: pd.DataFrame, indices: np.ndarray, outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract model-ready arrays for a named split."""
    subset = frame.iloc[indices]
    return (
        subset[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        subset["treatment"].to_numpy(dtype=np.int8),
        subset[outcome].to_numpy(dtype=np.int8),
    )
