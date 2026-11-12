#!/usr/bin/env python3
"""Run scale-free campaign-cost sensitivity analysis for the selected uplift policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from uplift_experiment.decision import optimal_policy_by_cost, profit_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=Path("reports/tables/policy_value.csv"))
    parser.add_argument("--sensitivity", type=Path, default=Path("reports/tables/profit_sensitivity.csv"))
    parser.add_argument("--optimal", type=Path, default=Path("reports/tables/optimal_policy_by_cost.csv"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/profit_sensitivity.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = pd.read_csv(args.policy)
    sensitivity = profit_sensitivity(policy)
    optimal = optimal_policy_by_cost(sensitivity)

    args.sensitivity.parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(args.sensitivity, index=False)
    optimal.to_csv(args.optimal, index=False)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for ratio, group in sensitivity.groupby(
        "cost_per_treatment_as_fraction_of_outcome_value", sort=True
    ):
        axis.plot(
            group["target_fraction"],
            group["net_value_units_per_10000_eligible"],
            marker="o",
            label=f"Cost/value = {ratio:.0%}",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        title="Targeting Depth Under Campaign-Cost Scenarios",
        xlabel="Fraction of eligible customers targeted",
        ylabel="Net value units per 10,000 eligible customers",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    print(
        optimal[
            [
                "cost_per_treatment_as_fraction_of_outcome_value",
                "target_fraction",
                "net_value_units_per_10000_eligible",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
