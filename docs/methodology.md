# Experiment analysis methodology

## Estimand

The primary estimand is the **intent-to-treat average risk difference**:

\[
\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0].
\]

Treatment assignment defines the contrast. `exposure` is post-treatment and is not used to redefine treatment, condition the main analysis, or build baseline targeting features.

## Unadjusted estimator

For each binary outcome (`visit` and `conversion`), the first estimator is the difference between treated and control sample means. Its standard error allows different outcome variances and sample sizes across arms.

This is the transparent benchmark: in a randomized experiment it does not require an outcome model.

## Covariate-adjusted estimator

The robustness and precision analysis uses Lin's fully interacted OLS adjustment:

- standardize and center the twelve pre-treatment features;
- include treatment, centered features, and every treatment-by-feature interaction;
- interpret the centered treatment coefficient as the adjusted sample ATE;
- use HC1 heteroskedasticity-robust standard errors.

The interactions matter because treatment allocation is approximately 85/15. A regression with common slopes can target an undesirable weighted effect when effects are heterogeneous and assignment is unequal.

## Power and minimum detectable effect

The project reports a prospective-style two-sided minimum detectable risk difference using:

- alpha: 5%;
- target power: 80%;
- observed control-group baseline rate;
- actual treatment and control sample sizes.

The MDE is a normal approximation. “Observed power” is included only as a descriptive diagnostic; the effect estimate and confidence interval remain the primary inferential quantities.

## Multiple outcomes

Visit and conversion form one confirmatory outcome family. Adjusted-estimator p-values are corrected using Holm's step-down procedure to control family-wise error without assuming independence.

## Interpretation boundaries

- The anonymized data supports methodological validation, not claims about a named customer segment or product.
- A statistically significant average effect does not prove every customer benefits.
- Uplift models will be evaluated only on held-out data using treatment-aware ranking metrics.
- Model selection will not use the final test set.
