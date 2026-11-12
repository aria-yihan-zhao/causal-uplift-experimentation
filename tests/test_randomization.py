import numpy as np
import pandas as pd

from uplift_experiment.data import EXPECTED_COLUMNS
from uplift_experiment.randomization import difference_in_proportions, experiment_audit


def randomized_frame(rows: int = 20_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.5, rows)
    features = rng.normal(size=(rows, 12))
    visit_probability = 0.10 + 0.03 * treatment
    conversion_probability = 0.01 + 0.005 * treatment
    payload = {f"f{i}": features[:, i] for i in range(12)}
    payload.update(
        treatment=treatment,
        conversion=rng.binomial(1, conversion_probability),
        visit=rng.binomial(1, visit_probability),
        exposure=treatment * rng.binomial(1, 0.8, rows),
    )
    return pd.DataFrame(payload, columns=EXPECTED_COLUMNS)


def test_difference_in_proportions_has_positive_visit_lift() -> None:
    result = difference_in_proportions(randomized_frame(), "visit")
    assert result["risk_difference"] > 0.02
    assert result["ci_low"] < result["risk_difference"] < result["ci_high"]


def test_randomization_audit_passes_balanced_synthetic_data() -> None:
    audit, balance = experiment_audit(randomized_frame())
    assert audit["balance"]["status"] == "PASS"
    assert balance["abs_smd"].max() < 0.10
    assert "exposure" not in balance["feature"].tolist()
