# Uplift model evaluation protocol

## Primary outcome

`visit` is the primary modeling outcome because its base rate is high enough to support stable ranking evaluation in the one-million-row analysis sample. `conversion` remains an average-effect outcome in the current milestone; its approximate 0.3% rate makes subgroup ranking substantially noisier.

## Locked split

The analysis uses one deterministic, stratified split:

- 60% training;
- 20% validation;
- 20% locked test.

Stratification preserves treatment-arm and primary-outcome proportions. Models are trained only on the training split. The selected algorithm is the one with maximum validation Qini. The test split is then used for the final, once-only comparison.

## Prespecified learners

- **S-Learner:** one gradient-boosted outcome model with treatment as an input, scored under both counterfactual treatment values.
- **T-Learner:** separate gradient-boosted outcome models for treated and control observations.
- **DR-Learner:** cross-fitted treated/control outcome models create a doubly robust pseudo-outcome using the known randomized propensity; a gradient-boosted regression model then estimates heterogeneous effects.

All three use the same twelve pre-treatment features and the same fixed training, validation, and test observations. Post-treatment `exposure` is excluded.

## Qini and AUUC

Customers are sorted by predicted uplift. At each targeting depth, cumulative incremental outcomes are estimated using the randomized-treatment propensity:

\[
\frac{T_iY_i}{e} - \frac{(1-T_i)Y_i}{1-e}.
\]

AUUC is the area under this cumulative uplift curve. Qini subtracts the area under random targeting. These metrics evaluate treatment-effect ranking; ordinary classification accuracy or ROC AUC cannot answer the same question.

## Policy value

For each model, the project estimates incremental visits when only the top 10%, 20%, 30%, 50%, or 100% of eligible customers are targeted. Values are expressed both per eligible customer and per 10,000 eligible customers.

This is an incremental-outcome analysis, not yet a profit estimate. Campaign cost, value per visit, and operational capacity will be introduced explicitly in the next decision-analysis milestone.

## Selection interpretation

The S-Learner narrowly achieved the highest validation Qini and was therefore selected under the prespecified rule. Its validation result was close to the DR-Learner's, so the project does not claim decisive statistical superiority without a paired uncertainty analysis.
