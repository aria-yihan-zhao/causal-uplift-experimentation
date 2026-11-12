"""Randomization diagnostics and descriptive intent-to-treat estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import FEATURE_COLUMNS, validate_data_contract


def standardized_mean_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute treated-minus-control standardized mean differences."""
    treated = frame.loc[frame["treatment"] == 1, FEATURE_COLUMNS]
    control = frame.loc[frame["treatment"] == 0, FEATURE_COLUMNS]
    mean_t = treated.mean()
    mean_c = control.mean()
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    smd = (mean_t - mean_c) / pooled_sd.replace(0, np.nan)
    return pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "treated_mean": mean_t.values,
            "control_mean": mean_c.values,
            "standardized_mean_difference": smd.values,
            "abs_smd": smd.abs().values,
        }
    ).sort_values("abs_smd", ascending=False, ignore_index=True)


def difference_in_proportions(
    frame: pd.DataFrame, outcome: str, confidence: float = 0.95
) -> dict[str, float]:
    """Estimate an unadjusted intent-to-treat risk difference and Wald CI."""
    treated = frame.loc[frame["treatment"] == 1, outcome]
    control = frame.loc[frame["treatment"] == 0, outcome]
    rate_t = float(treated.mean())
    rate_c = float(control.mean())
    difference = rate_t - rate_c
    standard_error = float(
        np.sqrt(rate_t * (1 - rate_t) / len(treated) + rate_c * (1 - rate_c) / len(control))
    )
    critical = float(norm.ppf(1 - (1 - confidence) / 2))
    return {
        "treated_rate": rate_t,
        "control_rate": rate_c,
        "risk_difference": difference,
        "relative_lift": difference / rate_c if rate_c else float("nan"),
        "standard_error": standard_error,
        "ci_low": difference - critical * standard_error,
        "ci_high": difference + critical * standard_error,
        "confidence_level": confidence,
    }


def experiment_audit(frame: pd.DataFrame, smd_warning_threshold: float = 0.10) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return a compact experiment audit and feature-level balance table."""
    contract = validate_data_contract(frame)
    balance = standardized_mean_differences(frame)
    max_abs_smd = float(balance["abs_smd"].max())
    n_treated = int(frame["treatment"].sum())
    n_control = int(len(frame) - n_treated)
    audit = {
        "contract": contract,
        "assignment": {
            "n_treated": n_treated,
            "n_control": n_control,
            "treatment_share": n_treated / len(frame),
        },
        "balance": {
            "warning_threshold_abs_smd": smd_warning_threshold,
            "max_abs_smd": max_abs_smd,
            "features_above_threshold": balance.loc[
                balance["abs_smd"] > smd_warning_threshold, "feature"
            ].tolist(),
            "status": "PASS" if max_abs_smd <= smd_warning_threshold else "REVIEW",
        },
        "descriptive_intent_to_treat": {
            outcome: difference_in_proportions(frame, outcome)
            for outcome in ["visit", "conversion"]
        },
        "guardrail": (
            "Exposure is post-treatment and was excluded from baseline balance checks. "
            "These unadjusted differences are descriptive; confirmatory specifications follow."
        ),
    }
    return audit, balance
