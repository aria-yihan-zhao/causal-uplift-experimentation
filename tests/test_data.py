import gzip

import pandas as pd
import pytest

from uplift_experiment.data import (
    EXPECTED_COLUMNS,
    canonical_url,
    sha256_file,
    validate_data_contract,
    verify_dataset_file,
)


def valid_frame() -> pd.DataFrame:
    rows = []
    for index in range(10):
        row = {f"f{i}": float(index + i) for i in range(12)}
        row.update(
            treatment=index % 2,
            conversion=int(index == 3),
            visit=int(index % 3 == 0),
            exposure=int(index % 2 == 1),
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_valid_data_contract() -> None:
    result = validate_data_contract(valid_frame())
    assert result["rows"] == 10
    assert result["missing_values"] == 0


def test_contract_rejects_nonbinary_treatment() -> None:
    frame = valid_frame()
    frame.loc[0, "treatment"] = 2
    with pytest.raises(ValueError, match="must be binary"):
        validate_data_contract(frame)


def test_contract_rejects_missing_values() -> None:
    frame = valid_frame()
    frame.loc[0, "f0"] = None
    with pytest.raises(ValueError, match="Missing values"):
        validate_data_contract(frame)


def test_canonical_url_removes_transient_parameters() -> None:
    url = "https://example.com/file.csv.gz?X-Amz-Signature=temporary#fragment"
    assert canonical_url(url) == "https://example.com/file.csv.gz"


def test_existing_dataset_requires_pinned_checksum(tmp_path) -> None:
    dataset = tmp_path / "dataset.csv.gz"
    with gzip.open(dataset, "wt", encoding="utf-8") as handle:
        handle.write(",".join(EXPECTED_COLUMNS) + "\n")
    digest = sha256_file(dataset)
    manifest = verify_dataset_file(dataset, expected_sha256=digest)
    assert manifest["sha256_verified"] is True
    assert "?" not in manifest["resolved_url"]

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_dataset_file(dataset, expected_sha256="0" * 64)
