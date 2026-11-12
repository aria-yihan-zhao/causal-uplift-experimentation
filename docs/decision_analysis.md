# Decision and unit-economics analysis

## Why the analysis is scale-free

The anonymized benchmark contains outcomes but no campaign cost or monetary value per visit. Assigning arbitrary dollars would create false precision. Instead, the analysis defines:

- one incremental visit = one outcome-value unit;
- treatment cost = a fraction of that outcome value.

For example, a 2% cost/value ratio means treating one customer costs 0.02 units when one incremental visit is worth 1.00 unit. A company can translate the results into dollars by multiplying all values by its own credible value per visit.

## Net-value equation

For a targeting fraction \(q\):

\[
\text{Net value}
= \text{incremental visits} \times \text{value per visit}
- q \times N \times \text{cost per treatment}.
\]

Policies include treating nobody, the top 10%, 20%, 30%, 50%, or everybody according to the validation-selected S-Learner.

## Optimal policies in the tested grid

| Treatment cost / visit value | Selected targeting depth | Net value units per 10,000 eligible |
|---:|---:|---:|
| 0% | 50% | 102.57 |
| 1% | 10% | 77.72 |
| 2% | 10% | 67.72 |
| 5% | 10% | 37.72 |
| 8% | 10% | 7.72 |
| 10% | 0% | 0.00 |

With zero treatment cost, targeting 50% marginally exceeded treating everyone in this test sample, suggesting that the excluded half contains little or negative incremental response. The difference is small and should not be treated as conclusive without policy-value uncertainty intervals.

The top-decile policy has an estimated break-even treatment cost equal to 8.77% of the value of one incremental visit. At the tested 10% cost/value ratio, treating nobody dominates all available targeting depths.

## Operational interpretation

- When outreach is nearly free, broad targeting can be reasonable.
- Even modest treatment cost makes uplift ranking economically important.
- A ranking model is not itself a deployment decision; value, cost, capacity, and uncertainty determine the cutoff.
- Production use would replace the scale-free inputs with finance-approved unit economics and add confidence bounds around policy value.
