"""Budget and unit-economics sensitivity analysis for targeting policies."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


REQUIRED_POLICY_COLUMNS = {
    "model",
    "target_fraction",
    "targeted_customers",
    "incremental_outcomes_per_eligible_customer",
    "incremental_outcomes_per_10000_eligible",
    "average_incremental_outcome_among_targeted",
    "selected_on_validation",
}


def profit_sensitivity(
    policy_table: pd.DataFrame,
    cost_value_ratios: Iterable[float] = (0.00, 0.01, 0.02, 0.05, 0.08, 0.10),
    outcome_value: float = 1.0,
    selected_only: bool = True,
) -> pd.DataFrame:
    """Translate incremental outcomes into scale-free net value.

    `outcome_value` defines one value unit per incremental outcome. A cost-value
    ratio of 0.02 therefore means each treatment costs 2% of the value assigned
    to one incremental outcome. This avoids inventing unavailable dollar inputs.
    """
    missing = REQUIRED_POLICY_COLUMNS.difference(policy_table.columns)
    if missing:
        raise ValueError(f"Policy table is missing columns: {sorted(missing)}")
    if outcome_value <= 0:
        raise ValueError("outcome_value must be positive.")

    working = policy_table.copy()
    if selected_only:
        working = working.loc[working["selected_on_validation"].astype(bool)].copy()
    if working.empty:
        raise ValueError("No policy rows are available for sensitivity analysis.")

    rows: list[dict[str, float | str | int | bool]] = []
    for model, model_policies in working.groupby("model", sort=False):
        selected_flag = bool(model_policies["selected_on_validation"].iloc[0])
        no_target = {
            "model": model,
            "target_fraction": 0.0,
            "targeted_customers": 0,
            "incremental_outcomes_per_eligible_customer": 0.0,
            "incremental_outcomes_per_10000_eligible": 0.0,
            "average_incremental_outcome_among_targeted": 0.0,
            "selected_on_validation": selected_flag,
        }
        candidates = pd.concat([pd.DataFrame([no_target]), model_policies], ignore_index=True)

        for ratio in cost_value_ratios:
            ratio = float(ratio)
            if ratio < 0:
                raise ValueError("Cost-value ratios cannot be negative.")
            for record in candidates.to_dict(orient="records"):
                target_fraction = float(record["target_fraction"])
                gross_value = (
                    float(record["incremental_outcomes_per_10000_eligible"])
                    * outcome_value
                )
                cost = target_fraction * 10_000 * ratio * outcome_value
                average_uplift = float(record["average_incremental_outcome_among_targeted"])
                rows.append(
                    {
                        **record,
                        "outcome_value_units": outcome_value,
                        "cost_per_treatment_as_fraction_of_outcome_value": ratio,
                        "gross_value_units_per_10000_eligible": gross_value,
                        "targeting_cost_units_per_10000_eligible": cost,
                        "net_value_units_per_10000_eligible": gross_value - cost,
                        "break_even_cost_fraction_of_outcome_value": (
                            average_uplift if target_fraction > 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def optimal_policy_by_cost(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Select the value-maximizing targeting depth for each model and cost."""
    grouping = ["model", "cost_per_treatment_as_fraction_of_outcome_value"]
    selected_rows = []
    for _, group in sensitivity.groupby(grouping, sort=True):
        selected_rows.append(group.loc[group["net_value_units_per_10000_eligible"].idxmax()])
    optimal = pd.DataFrame(selected_rows).reset_index(drop=True)
    return optimal.sort_values(grouping, ignore_index=True)
