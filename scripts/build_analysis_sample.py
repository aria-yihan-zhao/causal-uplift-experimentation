#!/usr/bin/env python3
"""Build an exact, reproducible sample from the full compressed CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from uplift_experiment.data import SOURCE_ROWS, build_exact_sample, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/criteo-uplift-v2.1.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/analysis_sample.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/tables/sample_manifest.json"))
    parser.add_argument("--sample-rows", type=int, default=1_000_000)
    parser.add_argument("--source-rows", type=int, default=SOURCE_ROWS)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_exact_sample(
        source=args.source,
        destination=args.output,
        sample_rows=args.sample_rows,
        seed=args.seed,
        source_rows=args.source_rows,
        chunksize=args.chunksize,
    )
    write_json(manifest, args.manifest)
    print(f"Wrote {manifest['sample_rows']:,} rows to {args.output}")
    print(f"Sample SHA-256: {manifest['sample_sha256']}")


if __name__ == "__main__":
    main()
