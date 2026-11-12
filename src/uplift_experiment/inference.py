"""Average-treatment-effect inference and experiment power diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import FEATURE_COLUMNS, validate_data_contract


def _normal_summary(effect: float, standard_error: float, confidence: float = 0.95) -> dict[str, float]:
    """Return a two-sided normal confidence interval and p-value."""
    if standard_error <= 0:
        raise ValueError("standard_error must be positive.")
    critical = float(norm.ppf(1 - (1 - confidence) / 2))
    z_score = effect / standard_error
    return {
        "effect": float(effect),
        "standard_error": float(standard_error),
        "z_score": float(z_score),
        "p_value": float(2 * norm.sf(abs(z_score))),
        "ci_low": float(effect - critical * standard_error),
        "ci_high": float(effect + critical * standard_error),
        "confidence_level": confidence,
    }


def unadjusted_ate(frame: pd.DataFrame, outcome: str, confidence: float = 0.95) -> dict[str, Any]:
    """Estimate the intent-to-treat difference in means with a robust analytic SE."""
    treated = frame.loc[frame["treatment"] == 1, outcome].to_numpy(dtype=float)
    control = frame.loc[frame["treatment"] == 0, outcome].to_numpy(dtype=float)
    effect = float(treated.mean() - control.mean())
    standard_error = float(
        np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    )
    result: dict[str, Any] = _normal_summary(effect, standard_error, confidence)
    result.update(
        {
            "estimator": "unadjusted_difference_in_means",
            "outcome": outcome,
            "n_treated": int(len(treated)),
            "n_control": int(len(control)),
            "treated_rate": float(treated.mean()),
            "control_rate": float(control.mean()),
        }
    )
    return result


def covariate_adjusted_ate(
    frame: pd.DataFrame,
    outcome: str,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate a Lin fully interacted ATE with an HC1 robust SE.

    Under randomized assignment, pre-treatment covariate adjustment is used for
    precision rather than identification. Features are standardized and centered,
    then interacted with treatment. Centering makes the treatment coefficient the
    covariate-adjusted sample ATE; interactions protect that estimand when treatment
    allocation is unequal and effects are heterogeneous.
    """
    validate_data_contract(frame)
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    means = features.mean(axis=0)
    scales = features.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    standardized = (features - means) / scales

    treatment = frame["treatment"].to_numpy(dtype=float)
    treatment_interactions = standardized * treatment[:, None]
    design = np.column_stack(
        [np.ones(len(frame)), treatment, standardized, treatment_interactions]
    )
    outcome_values = frame[outcome].to_numpy(dtype=float)

    coefficients, _, _, _ = np.linalg.lstsq(design, outcome_values, rcond=None)
    residuals = outcome_values - design @ coefficients
    inverse_xtx = np.linalg.inv(design.T @ design)
    meat = design.T @ (design * np.square(residuals)[:, None])
    n_rows, n_parameters = design.shape
    hc1_covariance = (n_rows / (n_rows - n_parameters)) * inverse_xtx @ meat @ inverse_xtx

    effect = float(coefficients[1])
    standard_error = float(np.sqrt(hc1_covariance[1, 1]))
    result: dict[str, Any] = _normal_summary(effect, standard_error, confidence)
    result.update(
        {
            "estimator": "lin_fully_interacted_ols_hc1",
            "outcome": outcome,
            "n_rows": int(n_rows),
            "covariates": FEATURE_COLUMNS,
            "treatment_interactions": FEATURE_COLUMNS,
        }
    )
    return result


def minimum_detectable_effect(
    baseline_rate: float,
    n_treated: int,
    n_control: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Approximate the two-sided MDE for a binary-outcome risk difference."""
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1.")
    if n_treated <= 0 or n_control <= 0:
        raise ValueError("Both treatment arms must contain observations.")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between 0 and 1.")
    critical = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    standard_error_null = np.sqrt(
        baseline_rate * (1 - baseline_rate) * (1 / n_treated + 1 / n_control)
    )
    return float(critical * standard_error_null)


def achieved_power(effect: float, standard_error: float, alpha: float = 0.05) -> float:
    """Return normal-approximation power at a specified effect and SE."""
    if standard_error <= 0:
        raise ValueError("standard_error must be positive.")
    critical = float(norm.ppf(1 - alpha / 2))
    noncentrality = abs(effect) / standard_error
    return float(norm.sf(critical - noncentrality) + norm.cdf(-critical - noncentrality))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise-error correction."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def ate_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    """Run two estimators, power diagnostics, and multiplicity control."""
    validate_data_contract(frame)
    outcomes: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}

    for outcome in ["visit", "conversion"]:
        unadjusted = unadjusted_ate(frame, outcome)
        adjusted = covariate_adjusted_ate(frame, outcome)
        raw_p_values[outcome] = adjusted["p_value"]
        mde = minimum_detectable_effect(
            baseline_rate=unadjusted["control_rate"],
            n_treated=unadjusted["n_treated"],
            n_control=unadjusted["n_control"],
        )
        outcomes[outcome] = {
            "unadjusted": unadjusted,
            "covariate_adjusted": adjusted,
            "power": {
                "alpha": 0.05,
                "target_power": 0.80,
                "minimum_detectable_risk_difference": mde,
                "observed_effect_over_mde": abs(unadjusted["effect"]) / mde,
                "normal_approximation_achieved_power": achieved_power(
                    unadjusted["effect"], unadjusted["standard_error"]
                ),
            },
        }

    adjusted_p_values = holm_adjust(raw_p_values)
    for outcome, adjusted_p_value in adjusted_p_values.items():
        outcomes[outcome]["covariate_adjusted"]["holm_adjusted_p_value"] = adjusted_p_value

    return {
        "estimand": "intent-to-treat average risk difference",
        "outcomes": outcomes,
        "multiplicity": {
            "method": "Holm family-wise-error correction across visit and conversion",
            "family_size": 2,
        },
        "interpretation_guardrails": [
            "Treatment assignment, not post-treatment exposure, defines the causal contrast.",
            "Power calculations use a normal approximation and the observed control rate.",
            "Covariate adjustment uses only anonymized pre-treatment features f0-f11.",
            "Observed power is descriptive; design decisions should rely on prospective MDE.",
        ],
    }
