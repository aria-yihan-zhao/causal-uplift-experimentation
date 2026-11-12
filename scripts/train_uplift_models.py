#!/usr/bin/env python3
"""Train prespecified uplift learners and evaluate on validation and locked test data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from uplift_experiment.data import write_json
from uplift_experiment.uplift import (
    DRLearner,
    SLearner,
    TLearner,
    frame_arrays,
    locked_split,
    policy_value_table,
    split_manifest,
    uplift_curve,
    uplift_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/analysis_sample.parquet"))
    parser.add_argument("--outcome", default="visit", choices=["visit", "conversion"])
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--comparison", type=Path, default=Path("reports/tables/uplift_model_comparison.csv"))
    parser.add_argument("--policy", type=Path, default=Path("reports/tables/policy_value.csv"))
    parser.add_argument("--split-manifest", type=Path, default=Path("reports/tables/split_manifest.json"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/uplift_curves.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    splits = locked_split(frame, outcome=args.outcome, seed=args.seed)
    write_json(split_manifest(frame, splits, args.outcome, args.seed), args.split_manifest)

    train_x, train_t, train_y = frame_arrays(frame, splits["train"], args.outcome)
    validation_x, validation_t, validation_y = frame_arrays(
        frame, splits["validation"], args.outcome
    )
    test_x, test_t, test_y = frame_arrays(frame, splits["test"], args.outcome)

    models = {
        "S-Learner": SLearner(args.seed, args.max_iter),
        "T-Learner": TLearner(args.seed, args.max_iter),
        "DR-Learner": DRLearner(args.seed, args.max_iter),
    }
    predictions: dict[str, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train_x, train_t, train_y)
        validation_score = model.predict_uplift(validation_x)
        test_score = model.predict_uplift(test_x)
        predictions[name] = {"validation": validation_score, "test": test_score}

        for split_name, y, t, score in [
            ("validation", validation_y, validation_t, validation_score),
            ("test", test_y, test_t, test_score),
        ]:
            metrics = uplift_metrics(y, t, score)
            comparison_rows.append(
                {
                    "model": name,
                    "split": split_name,
                    "outcome": args.outcome,
                    **metrics,
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    validation = comparison.loc[comparison["split"] == "validation"]
    selected_model = str(validation.sort_values("qini", ascending=False).iloc[0]["model"])
    comparison["selected_on_validation"] = comparison["model"].eq(selected_model)
    comparison["selection_rule"] = "maximum validation Qini"
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison, index=False)

    policy_tables = []
    for name in models:
        policy_tables.append(
            policy_value_table(
                test_y,
                test_t,
                predictions[name]["test"],
                model_name=name,
            )
        )
    policy = pd.concat(policy_tables, ignore_index=True)
    policy["selected_on_validation"] = policy["model"].eq(selected_model)
    policy.to_csv(args.policy, index=False)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for name in models:
        curve = uplift_curve(test_y, test_t, predictions[name]["test"])
        axis.plot(
            curve["population_fraction"],
            curve["cumulative_incremental_outcome"],
            label=name + (" (selected)" if name == selected_model else ""),
            linewidth=2.2 if name == selected_model else 1.5,
        )
    final_ate = uplift_metrics(test_y, test_t, predictions[selected_model]["test"])[
        "full_population_ipw_ate"
    ]
    axis.plot([0, 1], [0, final_ate], linestyle="--", color="gray", label="Random targeting")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(
        title=f"Locked-Test Uplift Curves — {args.outcome.title()} Outcome",
        xlabel="Fraction of eligible customers targeted",
        ylabel="Cumulative incremental outcomes per eligible customer",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    print(f"Selected on validation: {selected_model}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
