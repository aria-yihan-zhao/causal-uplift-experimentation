#!/usr/bin/env python3
"""Validate the analysis sample and write randomization diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uplift_experiment.data import write_json
from uplift_experiment.randomization import experiment_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/analysis_sample.parquet"))
    parser.add_argument("--audit-json", type=Path, default=Path("reports/tables/experiment_audit.json"))
    parser.add_argument("--balance-csv", type=Path, default=Path("reports/tables/covariate_balance.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.input)
    audit, balance = experiment_audit(frame)
    write_json(audit, args.audit_json)
    args.balance_csv.parent.mkdir(parents=True, exist_ok=True)
    balance.to_csv(args.balance_csv, index=False)
    print(f"Audit status: {audit['balance']['status']}")
    print(f"Maximum |SMD|: {audit['balance']['max_abs_smd']:.4f}")
    for outcome, result in audit["descriptive_intent_to_treat"].items():
        print(
            f"{outcome}: risk difference={result['risk_difference']:.6f}, "
            f"95% CI=({result['ci_low']:.6f}, {result['ci_high']:.6f})"
        )


if __name__ == "__main__":
    main()
