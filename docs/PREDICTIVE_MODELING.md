# Predictive modelling methodology

## Question and evidence boundary

Can public metadata in an Algorithmic Transparency Record distinguish records labelled `production` from records carrying another deployment-phase label?

This is an educational classification benchmark. The target is supplied by the publisher and is not an independently verified outcome. The model does not measure safety, fairness, effectiveness, legality, compliance, or governance maturity and must not be used to make deployment, procurement, regulatory, or organisational decisions.

## Data and target

The modelling table is the repository's dated official GOV.UK snapshot: 142 unique records, including 88 labelled production and 54 labelled another phase. The binary target is one when `phase == "production"` and zero otherwise.

The candidate feature set contains:

- organisation type, function, capability, region, and ATRS version;
- eight disclosure-presence indicators plus field-count and completeness measures; and
- for one challenger only, TF-IDF features from the public title and description.

Deployment phase, record ID, source URL, and publication date are excluded. This prevents direct target leakage and avoids a temporal feature that could dominate a small, evolving repository.

## Frozen experiment design

All candidates use the same `RepeatedStratifiedKFold` split design: five folds, five repeats, and random seed 42. Each validation record is unseen during its fold's model fit. Primary selection metric is mean ROC-AUC across the 25 validation folds; average precision, balanced accuracy, F1, and Brier score are secondary diagnostics.

The experiment sequence is:

1. prior-probability dummy baseline;
2. regularised metadata logistic regression;
3. TF-IDF plus metadata logistic regression;
4. balanced random forest;
5. gradient-boosted trees; and
6. gradient-boosted trees with sigmoid probability calibration.

The selection rule was fixed before looking at results: select the highest mean ROC-AUC, but prefer the lower-complexity candidate when it is within 0.01 of the best. This guards against treating a negligible score difference as evidence for a harder-to-explain model.

## Results

TF-IDF plus logistic regression had the highest fold-level mean ROC-AUC (0.690), while metadata logistic regression scored 0.682. The difference was 0.008, so the simplicity rule selected metadata logistic regression. Its mean fold-level ROC-AUC standard deviation was 0.083, underscoring the uncertainty in this small dataset.

Averaging the five repeated out-of-fold probabilities for each record produced ROC-AUC 0.693, average precision 0.793, balanced accuracy 0.620, F1 0.683, and Brier score 0.234. The aggregate score is a descriptive cross-validated estimate, not a test-set or external-validation result.

## Reproducibility and audit trail

- `results/model_experiments.tsv` records the primary score and keep/discard status for every candidate.
- `results/model_experiments.json` records all fold-level summary metrics.
- `results/oof_predictions.csv` makes record-level cross-validated predictions reviewable.
- `results/feature_effects.csv` provides signed logistic coefficients labelled as associations, not causal explanations.
- `results/model_metrics.json` records the input SHA-256, validation scheme, library version, and target definition.
- `models/MODEL_CARD.md` states intended and prohibited uses.

Install the optional dependencies and rerun with:

```bash
python -m pip install -e ".[ml]"
python -m atrs_audit.modeling --input data/processed/atrs_audit.csv --output-root .
```

## Limitations

1. The sample of 142 is too small for a production classifier or stable subgroup assessment.
2. Cross-validation reuses one snapshot and is not independent external or temporal validation.
3. ATRS versions and publishing practice changed over time, creating structural shift.
4. Organisation and wording features can act as proxies for publisher-specific conventions.
5. The binary target groups several distinct non-production phases together.
6. Coefficients are conditional associations and should not be interpreted causally.
7. A serialized `joblib` model should be loaded only from a trusted checkout; Python model artifacts are not a safe interchange format for untrusted files.

## Technical references

- [RepeatedStratifiedKFold — scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)
- [Probability calibration — scikit-learn user guide](https://scikit-learn.org/stable/modules/calibration.html)
- [Ensemble methods and gradient boosting — scikit-learn user guide](https://scikit-learn.org/stable/modules/ensemble.html)
