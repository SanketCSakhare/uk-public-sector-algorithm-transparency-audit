# Model card: ATRS production-phase classifier

## Summary

This exploratory classifier estimates whether a published UK Algorithmic Transparency Record is labelled **Production** rather than another deployment phase. The selected approach is **Regularised logistic regression**.

It is a portfolio demonstration of reproducible predictive modelling. It must not be used to assess safety, compliance, organisational capability, procurement, or whether an algorithmic tool should enter production.

## Training data

- Source: official GOV.UK Algorithmic Transparency Records snapshot
- Records: 142
- Positive class: 88 production records
- Negative class: 54 non-production records
- Publication range: 2022-07-06 to 2026-08-13
- Generated: 2026-08-20T23:04:47.196307+00:00

## Target and features

- Target: `1 when phase == 'production'; 0 otherwise`
- Selected model: metadata-only regularised logistic regression
- Text: public record title and description (evaluated only in the TF-IDF challenger)
- Metadata: organisation type, function, capability, region, ATRS version (used by selected model)
- Disclosure features: eight previously documented disclosure-presence indicators and field-count measures
- Explicitly excluded: deployment phase itself, record ID, source URL, and publication date

## Validation

Five-fold stratified cross-validation repeated five times with random seed 42. Primary metric: ROC-AUC. The selected model's aggregated out-of-fold metrics are:

- ROC-AUC: 0.693
- Average precision: 0.793
- Balanced accuracy at 0.5: 0.620
- F1 at 0.5: 0.683
- Brier score: 0.234

## Intended use

- Demonstrating a complete small-data classification workflow
- Comparing linear, bagged-tree, boosted-tree, and calibrated approaches
- Teaching cross-validation, leakage control, reproducibility, and model documentation

## Prohibited or inappropriate use

- Predicting deployment decisions for real public-sector systems
- Ranking organisations or inferring governance maturity
- Safety, fairness, legal, or regulatory assessment
- Automated decisions about people, suppliers, or public bodies

## Limitations

- Only 142 records are available, so model comparisons have high variance.
- The target is a publisher-supplied status label, not an independent outcome.
- Records span ATRS versions and publication years, creating structural and temporal shift.
- Public text can encode organisation-specific wording and process conventions.
- Repeated cross-validation reduces split sensitivity but does not create new independent evidence.
- Coefficients and feature effects are associations, not causal explanations.
- The serialized joblib artifact must be loaded only from a trusted checkout.

## Reproducibility

Run `python -m atrs_audit.modeling --input data/processed/atrs_audit.csv --output-root .` after installing `.[ml]`. Review `results/model_experiments.tsv`, `results/model_metrics.json`, and `results/oof_predictions.csv` before citing performance.
