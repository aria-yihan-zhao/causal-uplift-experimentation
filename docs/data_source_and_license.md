# Data source, version, and license

## Source

- Provider: Criteo AI Lab
- Dataset: Criteo Uplift Prediction Dataset, corrected/unbiased v2.1
- Official page: <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>
- Verified Criteo organization mirror: <https://huggingface.co/datasets/criteo/criteo-uplift>
- Downloaded file: `criteo-research-uplift-v2.1.csv.gz`
- Expected bytes: `311422618`
- Expected SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- Historical download redirect: <https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz>
- Reference paper: Diemert et al. (2018), *A Large Scale Benchmark for Uplift Modeling*

The corrected dataset contains 13,979,592 rows, twelve anonymized dense features (`f0`–`f11`), treatment assignment, conversion, visit, and effective exposure indicators.

The historical Criteo redirect was unavailable during the November 12, 2026 build. The downloader therefore uses the dataset published by the verified `criteo` organization on Hugging Face and refuses the file unless its SHA-256 matches the documented v2.1 digest above.

## Why v2.1 is required

Criteo documents an erratum for the original release: non-uniform incrementality across advertisers created a leak because advertiser identity could be partially inferred from feature distributions. This project therefore uses only the corrected/unbiased release.

## Causal-variable roles

- `f0`–`f11`: anonymized pre-treatment covariates
- `treatment`: randomized assignment indicator
- `visit`, `conversion`: binary outcomes
- `exposure`: whether a treated user was effectively exposed

`exposure` occurs after assignment. It must not be used as a baseline covariate in the randomization audit or as an ordinary feature in intent-to-treat targeting models.

## License and repository policy

Criteo publishes the data under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Users must review and comply with the official terms, including attribution, non-commercial use, and share-alike requirements.

This repository does **not** redistribute the raw data or derived row-level samples. The download script retrieves the file from the verified publisher mirror and records an integrity manifest. Only non-row-level aggregate audit statistics are committed, with clear attribution.

The repository's original source code and methodology documentation are separately licensed under MIT. The MIT license does not apply to the dataset. Aggregate tables, figures, and summaries under `reports/` are shared under CC BY-NC-SA 4.0; see [`reports/LICENSE.md`](../reports/LICENSE.md).
