#!/usr/bin/env python3
"""Download the corrected Criteo uplift dataset and record its integrity digest."""

from __future__ import annotations

import argparse
from pathlib import Path

from uplift_experiment.data import (
    EXPECTED_SHA256,
    SOURCE_URL,
    download_file,
    verify_dataset_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--output", type=Path, default=Path("data/raw/criteo-uplift-v2.1.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/tables/download_manifest.json"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        manifest = verify_dataset_file(
            args.output,
            expected_sha256=EXPECTED_SHA256,
            source_url=args.url,
        )
        write_json(manifest, args.manifest)
        print(f"Verified existing file: {args.output}")
        print(f"SHA-256: {manifest['sha256']}")
        return
    manifest = download_file(args.url, args.output, expected_sha256=EXPECTED_SHA256)
    write_json(manifest, args.manifest)
    print(f"Downloaded {manifest['bytes']:,} bytes to {args.output}")
    print(f"SHA-256: {manifest['sha256']}")


if __name__ == "__main__":
    main()
