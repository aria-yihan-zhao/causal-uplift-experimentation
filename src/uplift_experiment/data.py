"""Data download, validation, and deterministic sampling utilities."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests

FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["treatment", "conversion", "visit", "exposure"]
BINARY_COLUMNS = ["treatment", "conversion", "visit", "exposure"]
SOURCE_ROWS = 13_979_592
SOURCE_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
LEGACY_SOURCE_URL = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
EXPECTED_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_url(url: str) -> str:
    """Remove transient query parameters and fragments from a source URL."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def verify_dataset_file(
    path: Path,
    expected_sha256: str,
    source_url: str = SOURCE_URL,
) -> dict[str, Any]:
    """Verify an existing dataset against its pinned digest and schema."""
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}."
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    if header != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected CSV header: {header}")
    return {
        "source_url": canonical_url(source_url),
        "resolved_url": canonical_url(source_url),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_mode": "existing_file",
        "destination": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "sha256_verified": True,
        "expected_source_rows": SOURCE_ROWS,
        "columns": EXPECTED_COLUMNS,
    }


def download_file(
    url: str,
    destination: Path,
    timeout: int = 60,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Stream a remote file to disk and return an integrity manifest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        digest = hashlib.sha256()
        byte_count = 0
        with temporary.open("wb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                digest.update(block)
                byte_count += len(block)
        resolved_url = canonical_url(response.url)
        expected_bytes = response.headers.get("Content-Length")

    if expected_bytes is not None and byte_count != int(expected_bytes):
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Incomplete download: expected {expected_bytes} bytes, received {byte_count}."
        )

    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}."
        )

    with gzip.open(temporary, "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    if header != EXPECTED_COLUMNS:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Unexpected CSV header: {header}")

    temporary.replace(destination)
    return {
        "source_url": canonical_url(url),
        "resolved_url": resolved_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_mode": "downloaded",
        "destination": str(destination),
        "bytes": byte_count,
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "sha256_verified": expected_sha256 is None or observed_sha256 == expected_sha256,
        "expected_source_rows": SOURCE_ROWS,
        "columns": EXPECTED_COLUMNS,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a deterministic, human-readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_data_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate schema, missingness, finite features, and binary indicators."""
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Column contract failed. Expected {EXPECTED_COLUMNS}, got {list(frame.columns)}."
        )
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame.isna().any().any():
        missing = frame.isna().sum()
        raise ValueError(f"Missing values detected: {missing[missing > 0].to_dict()}")

    feature_values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Feature matrix contains non-finite values.")

    binary_values: dict[str, list[int]] = {}
    for column in BINARY_COLUMNS:
        observed = sorted(pd.unique(frame[column]).tolist())
        if not set(observed).issubset({0, 1}):
            raise ValueError(f"{column} must be binary; observed {observed}.")
        binary_values[column] = [int(value) for value in observed]

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "binary_values": binary_values,
        "missing_values": 0,
        "finite_features": True,
    }


def build_exact_sample(
    source: Path,
    destination: Path,
    sample_rows: int,
    seed: int,
    source_rows: int = SOURCE_ROWS,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    """Select an exact seeded sample by global row index while streaming the CSV."""
    if not 0 < sample_rows <= source_rows:
        raise ValueError("sample_rows must be between 1 and source_rows.")

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(source_rows, size=sample_rows, replace=False))
    pieces: list[pd.DataFrame] = []
    offset = 0

    for chunk in pd.read_csv(source, chunksize=chunksize):
        upper = offset + len(chunk)
        left = np.searchsorted(selected, offset, side="left")
        right = np.searchsorted(selected, upper, side="left")
        if right > left:
            local_positions = selected[left:right] - offset
            pieces.append(chunk.iloc[local_positions])
        offset = upper

    if offset != source_rows:
        raise ValueError(f"Source row count changed: expected {source_rows}, observed {offset}.")

    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) != sample_rows:
        raise RuntimeError(f"Sampling failed: expected {sample_rows}, got {len(sample)}.")

    contract = validate_data_contract(sample)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(destination, index=False)
    return {
        "source": str(source),
        "destination": str(destination),
        "source_rows": source_rows,
        "sample_rows": sample_rows,
        "seed": seed,
        "selection_method": "exact seeded uniform sample of global row indices without replacement",
        "sample_sha256": sha256_file(destination),
        "contract": contract,
    }
