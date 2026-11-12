#!/usr/bin/env python3
"""Estimate average treatment effects and experiment power."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uplift_experiment.data import write_json
from uplift_experiment.inference import ate_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/analysis_sample.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/tables/ate_analysis.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    analysis = ate_analysis(frame)
    write_json(analysis, args.output)

    for outcome, result in analysis["outcomes"].items():
        unadjusted = result["unadjusted"]
        adjusted = result["covariate_adjusted"]
        power = result["power"]
        print(
            f"{outcome}: unadjusted={unadjusted['effect']:.6f}, "
            f"adjusted={adjusted['effect']:.6f}, "
            f"adjusted 95% CI=({adjusted['ci_low']:.6f}, {adjusted['ci_high']:.6f}), "
            f"MDE={power['minimum_detectable_risk_difference']:.6f}"
        )


if __name__ == "__main__":
    main()
