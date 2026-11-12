# Executive summary

## Objective

Determine whether a marketing treatment creates incremental visits and conversions, identify customers whose behavior is most likely to change because of treatment, and choose a targeting depth under budget and unit-economics constraints.

## Data and integrity controls

The project uses the corrected Criteo Uplift v2.1 randomized-incrementality dataset. A deterministic one-million-row sample was drawn from 13,979,592 records after verifying the source file's pinned SHA-256 digest. Raw and row-level derived data remain outside Git.

The experiment audit found an 85.03% treatment share and a maximum absolute standardized mean difference of 0.0505 across twelve pre-treatment features, below the prespecified 0.10 review threshold. Post-treatment exposure was excluded from baseline adjustment and targeting features.

## Average causal effects

Both transparent difference-in-means estimates and Lin fully interacted covariate adjustment were reported. After covariate adjustment and Holm correction across two outcomes:

- visit increased by 0.8067 percentage points (95% CI: 0.7111 to 0.9022 pp);
- conversion increased by 0.0960 percentage points (95% CI: 0.0707 to 0.1214 pp).

The adjusted effects were smaller than the raw estimates, so the project retains both specifications rather than selecting the more favorable number.

## Heterogeneous-effect modeling

S-Learner, T-Learner, and cross-fitted Doubly Robust Learner models were trained on a fixed 60% split. Model selection used only 20% validation data; the remaining 20% was locked for final evaluation.

The S-Learner narrowly achieved the highest validation Qini and also produced the highest locked-test Qini (0.004683). The validation margin over the DR-Learner was small, so the result is treated as a prespecified selection outcome rather than decisive algorithmic superiority.

## Targeting decision

On locked test data, treating only the top 10% of customers ranked by predicted uplift captured approximately 85.8% of the incremental visits obtained by treating the full eligible population.

Because monetary inputs are unavailable, policy economics are expressed as treatment cost divided by the value of one incremental visit. The optimal tested policy was:

- target 50% when treatment is free;
- target 10% when treatment cost is 1%–8% of visit value;
- target nobody at the tested 10% cost/value ratio.

The top-decile policy's estimated break-even treatment cost was 8.77% of one incremental visit's value.

## Recommendation and limitations

Use uplift ranking, rather than response probability, when campaign cost or capacity constrains targeting. Before production deployment, replace scale-free inputs with approved unit economics, add paired uncertainty intervals for model and policy differences, monitor population drift, and validate the decision rule in a new randomized holdout.

The dataset is anonymized, so the analysis demonstrates causal-modeling and decision methodology rather than claims about a particular product, customer group, or market.
