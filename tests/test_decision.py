import pandas as pd

from uplift_experiment.decision import optimal_policy_by_cost, profit_sensitivity


def example_policy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["S-Learner", "S-Learner"],
            "target_fraction": [0.1, 1.0],
            "targeted_customers": [100, 1000],
            "incremental_outcomes_per_eligible_customer": [0.008, 0.010],
            "incremental_outcomes_per_10000_eligible": [80.0, 100.0],
            "average_incremental_outcome_among_targeted": [0.08, 0.01],
            "selected_on_validation": [True, True],
        }
    )


def test_zero_cost_prefers_larger_gross_value() -> None:
    sensitivity = profit_sensitivity(example_policy(), cost_value_ratios=[0.0])
    optimal = optimal_policy_by_cost(sensitivity)
    assert optimal.iloc[0]["target_fraction"] == 1.0
    assert optimal.iloc[0]["net_value_units_per_10000_eligible"] == 100.0


def test_high_cost_can_choose_no_treatment() -> None:
    sensitivity = profit_sensitivity(example_policy(), cost_value_ratios=[0.20])
    optimal = optimal_policy_by_cost(sensitivity)
    assert optimal.iloc[0]["target_fraction"] == 0.0
    assert optimal.iloc[0]["net_value_units_per_10000_eligible"] == 0.0


def test_break_even_cost_matches_average_uplift() -> None:
    sensitivity = profit_sensitivity(example_policy(), cost_value_ratios=[0.01])
    top_decile = sensitivity.loc[sensitivity["target_fraction"] == 0.1].iloc[0]
    assert top_decile["break_even_cost_fraction_of_outcome_value"] == 0.08
