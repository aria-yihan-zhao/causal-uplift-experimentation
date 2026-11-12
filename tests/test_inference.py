import numpy as np
import pandas as pd

from uplift_experiment.data import EXPECTED_COLUMNS
from uplift_experiment.inference import (
    achieved_power,
    covariate_adjusted_ate,
    holm_adjust,
    minimum_detectable_effect,
    unadjusted_ate,
)


def synthetic_experiment(rows: int = 40_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 12))
    treatment = rng.binomial(1, 0.5, size=rows)
    probability = np.clip(0.15 + 0.04 * treatment + 0.02 * features[:, 0], 0.01, 0.99)
    payload = {f"f{i}": features[:, i] for i in range(12)}
    payload.update(
        treatment=treatment,
        conversion=rng.binomial(1, 0.01, size=rows),
        visit=rng.binomial(1, probability),
        exposure=treatment * rng.binomial(1, 0.8, size=rows),
    )
    return pd.DataFrame(payload, columns=EXPECTED_COLUMNS)


def test_unadjusted_and_adjusted_estimators_recover_effect() -> None:
    frame = synthetic_experiment()
    unadjusted = unadjusted_ate(frame, "visit")
    adjusted = covariate_adjusted_ate(frame, "visit")
    assert 0.025 < unadjusted["effect"] < 0.055
    assert 0.025 < adjusted["effect"] < 0.055
    assert adjusted["ci_low"] < adjusted["effect"] < adjusted["ci_high"]


def test_power_helpers_behave_monotonically() -> None:
    small_mde = minimum_detectable_effect(0.05, 100_000, 20_000)
    large_mde = minimum_detectable_effect(0.05, 10_000, 2_000)
    assert small_mde < large_mde
    assert achieved_power(0.02, 0.002) > achieved_power(0.005, 0.002)


def test_holm_adjustment_controls_order() -> None:
    adjusted = holm_adjust({"visit": 0.01, "conversion": 0.04, "retention": 0.20})
    assert adjusted["visit"] == 0.03
    assert adjusted["conversion"] == 0.08
    assert adjusted["retention"] == 0.20
