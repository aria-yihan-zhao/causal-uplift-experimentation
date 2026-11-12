import numpy as np
import pandas as pd

from uplift_experiment.data import EXPECTED_COLUMNS
from uplift_experiment.uplift import (
    locked_split,
    policy_value_table,
    uplift_metrics,
)


def split_frame(rows: int = 10_000, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = rng.binomial(1, 0.7, rows)
    visit = rng.binomial(1, 0.08 + 0.03 * treatment, rows)
    payload = {f"f{i}": rng.normal(size=rows) for i in range(12)}
    payload.update(
        treatment=treatment,
        conversion=rng.binomial(1, 0.01, rows),
        visit=visit,
        exposure=treatment * rng.binomial(1, 0.8, rows),
    )
    return pd.DataFrame(payload, columns=EXPECTED_COLUMNS)


def test_locked_split_is_complete_disjoint_and_deterministic() -> None:
    frame = split_frame()
    first = locked_split(frame)
    second = locked_split(frame)
    combined = np.concatenate(list(first.values()))
    assert len(combined) == len(frame)
    assert len(np.unique(combined)) == len(frame)
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_perfect_effect_ranking_beats_reversed_ranking() -> None:
    rng = np.random.default_rng(4)
    rows = 200_000
    true_effect = rng.uniform(0, 0.20, rows)
    treatment = rng.binomial(1, 0.5, rows)
    probability = 0.05 + treatment * true_effect
    outcome = rng.binomial(1, probability)
    perfect = uplift_metrics(outcome, treatment, true_effect)
    reversed_result = uplift_metrics(outcome, treatment, -true_effect)
    assert perfect["qini"] > 0
    assert perfect["qini"] > reversed_result["qini"]


def test_policy_value_table_has_requested_fractions() -> None:
    frame = split_frame()
    score = frame["f0"].to_numpy()
    table = policy_value_table(
        frame["visit"].to_numpy(),
        frame["treatment"].to_numpy(),
        score,
        model_name="test",
    )
    assert table["target_fraction"].tolist() == [0.10, 0.20, 0.30, 0.50, 1.00]
    assert np.isfinite(table["incremental_outcomes_per_eligible_customer"]).all()
