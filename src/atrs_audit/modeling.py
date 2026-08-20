"""Reproducible predictive-modelling experiments for the ATRS snapshot.

The model predicts whether a published record is in the production phase. It is
an educational portfolio experiment, not a tool for assessing safety,
compliance, organisational maturity, or future deployment decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42
PRIMARY_METRIC = "roc_auc"
TARGET_DEFINITION = "1 when phase == 'production'; 0 otherwise"

CATEGORICAL_COLUMNS = (
    "organisation_type",
    "function",
    "capability",
    "region",
    "atrs_version",
)
NUMERIC_COLUMNS = (
    "accountability",
    "human_oversight",
    "contestability",
    "impact_assessment",
    "risks_mitigations",
    "model_performance",
    "data_governance",
    "monitoring_maintenance",
    "field_count",
    "substantive_field_count",
    "field_completeness_ratio",
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    approach: str
    complexity: int
    description: str
    estimator: BaseEstimator


def load_records(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The modelling dataset is empty")
    required = {"record_id", "title", "description", "phase", *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"Missing required modelling columns: {missing}")
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("record_id must be unique at the modelling grain")
    target = np.asarray([int(row["phase"] == "production") for row in rows], dtype=int)
    if set(target) != {0, 1}:
        raise ValueError("The production target must contain both classes")
    return np.asarray(rows, dtype=object), target


def select_text(records: Sequence[dict[str, str]]) -> list[str]:
    return [f"{row.get('title', '')} {row.get('description', '')}".strip() for row in records]


def select_metadata(records: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in records:
        features: dict[str, Any] = {}
        for column in CATEGORICAL_COLUMNS:
            values = [value for value in row.get(column, "").split("|") if value]
            if not values:
                features[f"{column}=missing"] = 1.0
            for value in values:
                features[f"{column}={value}"] = 1.0
        for column in NUMERIC_COLUMNS:
            raw = row.get(column, "")
            if str(raw).casefold() in {"true", "false"}:
                features[column] = float(str(raw).casefold() == "true")
            else:
                features[column] = float(raw or 0.0)
        output.append(features)
    return output


def metadata_pipeline(classifier: BaseEstimator, *, dense: bool) -> Pipeline:
    return Pipeline(
        [
            (
                "metadata",
                Pipeline(
                    [
                        (
                            "select",
                            FunctionTransformer(select_metadata, validate=False),
                        ),
                        ("vectorize", DictVectorizer(sparse=not dense)),
                        *([] if dense else [("scale", StandardScaler(with_mean=False))]),
                    ]
                ),
            ),
            ("classifier", classifier),
        ]
    )


def text_metadata_logistic() -> Pipeline:
    text = Pipeline(
        [
            ("select", FunctionTransformer(select_text, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=2,
                    max_df=0.95,
                    ngram_range=(1, 2),
                    max_features=1_200,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    metadata = Pipeline(
        [
            ("select", FunctionTransformer(select_metadata, validate=False)),
            ("vectorize", DictVectorizer()),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    return Pipeline(
        [
            ("features", FeatureUnion([("text", text), ("metadata", metadata)])),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_experiments() -> list[Experiment]:
    logistic = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=2_000,
        random_state=RANDOM_STATE,
    )
    forest = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    gradient = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.035,
        max_depth=2,
        min_samples_leaf=5,
        subsample=0.85,
        random_state=RANDOM_STATE,
    )
    gradient_pipeline = metadata_pipeline(gradient, dense=True)
    return [
        Experiment(
            "exp_00_dummy",
            "Prior-probability baseline",
            0,
            "Predicts the training-fold class prior and uses no record features.",
            DummyClassifier(strategy="prior"),
        ),
        Experiment(
            "exp_01_metadata_logistic",
            "Regularised logistic regression",
            1,
            "Interpretable linear baseline using public categorical and disclosure metadata.",
            metadata_pipeline(logistic, dense=False),
        ),
        Experiment(
            "exp_02_text_metadata_logistic",
            "TF-IDF plus logistic regression",
            2,
            "Adds title and description TF-IDF features to the metadata baseline.",
            text_metadata_logistic(),
        ),
        Experiment(
            "exp_03_random_forest",
            "Balanced random forest",
            3,
            "Non-linear bagged-tree challenger on public metadata.",
            metadata_pipeline(forest, dense=True),
        ),
        Experiment(
            "exp_04_gradient_boosting",
            "Gradient-boosted trees",
            4,
            "Modern boosted-tree challenger tuned conservatively for the small sample.",
            gradient_pipeline,
        ),
        Experiment(
            "exp_05_calibrated_gradient_boosting",
            "Calibrated gradient-boosted trees",
            5,
            "Gradient boosting with cross-validated sigmoid probability calibration.",
            CalibratedClassifierCV(
                estimator=gradient_pipeline,
                method="sigmoid",
                cv=3,
                ensemble=True,
                n_jobs=1,
            ),
        ),
    ]


def validation_scheme() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )


def evaluate_experiments(
    experiments: Sequence[Experiment], X: np.ndarray, y: np.ndarray
) -> list[dict[str, Any]]:
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "neg_brier": "neg_brier_score",
    }
    results: list[dict[str, Any]] = []
    for experiment in experiments:
        row: dict[str, Any] = {
            "experiment_id": experiment.experiment_id,
            "approach": experiment.approach,
            "complexity": experiment.complexity,
            "description": experiment.description,
            "status": "evaluated",
        }
        try:
            scores = cross_validate(
                experiment.estimator,
                X,
                y,
                scoring=scoring,
                cv=validation_scheme(),
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            for metric in scoring:
                values = scores[f"test_{metric}"]
                sign = -1.0 if metric == "neg_brier" else 1.0
                clean_metric = "brier" if metric == "neg_brier" else metric
                row[f"{clean_metric}_mean"] = round(float(np.mean(values) * sign), 4)
                row[f"{clean_metric}_std"] = round(float(np.std(values)), 4)
        except Exception as exc:  # Preserve failed experiments instead of hiding them.
            row["status"] = "crash"
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    return results


def select_experiment(
    experiments: Sequence[Experiment], results: list[dict[str, Any]]
) -> Experiment:
    successful = [row for row in results if row["status"] == "evaluated" and row["complexity"] > 0]
    if not successful:
        raise RuntimeError("No non-dummy modelling experiment completed")
    best_auc = max(float(row["roc_auc_mean"]) for row in successful)
    practical_ties = [row for row in successful if float(row["roc_auc_mean"]) >= best_auc - 0.01]
    chosen_row = min(practical_ties, key=lambda row: int(row["complexity"]))
    for row in results:
        if row["status"] == "crash":
            continue
        row["status"] = "keep" if row["experiment_id"] == chosen_row["experiment_id"] else "discard"
    return next(item for item in experiments if item.experiment_id == chosen_row["experiment_id"])


def out_of_fold_predictions(
    estimator: BaseEstimator, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    probabilities = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    for train_index, test_index in validation_scheme().split(X, y):
        fitted = clone(estimator).fit(X[train_index], y[train_index])
        probabilities[test_index] += fitted.predict_proba(X[test_index])[:, 1]
        counts[test_index] += 1
    if np.any(counts == 0):
        raise RuntimeError("Every record must receive an out-of-fold prediction")
    return probabilities / counts


def prediction_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y, probabilities)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 4),
        "f1": round(float(f1_score(y, predicted)), 4),
        "brier_score": round(float(brier_score_loss(y, probabilities)), 4),
        "threshold": 0.5,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def linear_feature_effects(model: BaseEstimator) -> tuple[str, list[dict[str, Any]]]:
    """Return coefficients from the interpretable text+metadata challenger."""
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        return "", []
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return "", []
    names: list[str] = []
    if "features" in model.named_steps:
        union = model.named_steps["features"]
        transformers = dict(union.transformer_list)
        text_names = transformers["text"].named_steps["tfidf"].get_feature_names_out()
        metadata_names = transformers["metadata"].named_steps["vectorize"].get_feature_names_out()
        names.extend(f"text:{name}" for name in text_names)
        names.extend(f"metadata:{name}" for name in metadata_names)
    elif "metadata" in model.named_steps:
        metadata_names = model.named_steps["metadata"].named_steps["vectorize"].get_feature_names_out()
        names.extend(f"metadata:{name}" for name in metadata_names)
    coefficients = classifier.coef_[0]
    if len(names) != len(coefficients):
        return "", []
    rows = [
        {
            "feature": name,
            "coefficient": round(float(value), 6),
            "direction": "production" if value > 0 else "non-production",
        }
        for name, value in zip(names, coefficients)
    ]
    rows.sort(key=lambda row: abs(float(row["coefficient"])), reverse=True)
    return "logistic coefficient; association only, not causal importance", rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_log(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "experiment_id",
        "approach",
        "metric_name",
        "metric_value",
        "complexity",
        "status",
        "description",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "experiment_id": result["experiment_id"],
                    "approach": result["approach"],
                    "metric_name": PRIMARY_METRIC,
                    "metric_value": result.get("roc_auc_mean", ""),
                    "complexity": result["complexity"],
                    "status": result["status"],
                    "description": result.get("description", result.get("error", "")),
                }
            )


def build_model_card(
    selected: Experiment,
    profile: dict[str, Any],
    metrics: dict[str, Any],
    generated_at: str,
) -> str:
    return f"""# Model card: ATRS production-phase classifier

## Summary

This exploratory classifier estimates whether a published UK Algorithmic Transparency Record is labelled **Production** rather than another deployment phase. The selected approach is **{selected.approach}**.

It is a portfolio demonstration of reproducible predictive modelling. It must not be used to assess safety, compliance, organisational capability, procurement, or whether an algorithmic tool should enter production.

## Training data

- Source: official GOV.UK Algorithmic Transparency Records snapshot
- Records: {profile['record_count']}
- Positive class: {profile['production_records']} production records
- Negative class: {profile['non_production_records']} non-production records
- Publication range: {profile['date_min']} to {profile['date_max']}
- Generated: {generated_at}

## Target and features

- Target: `{TARGET_DEFINITION}`
- Selected model: metadata-only regularised logistic regression
- Text: public record title and description (evaluated only in the TF-IDF challenger)
- Metadata: organisation type, function, capability, region, ATRS version (used by selected model)
- Disclosure features: eight previously documented disclosure-presence indicators and field-count measures
- Explicitly excluded: deployment phase itself, record ID, source URL, and publication date

## Validation

Five-fold stratified cross-validation repeated five times with random seed {RANDOM_STATE}. Primary metric: ROC-AUC. The selected model's aggregated out-of-fold metrics are:

- ROC-AUC: {metrics['roc_auc']:.3f}
- Average precision: {metrics['average_precision']:.3f}
- Balanced accuracy at 0.5: {metrics['balanced_accuracy']:.3f}
- F1 at 0.5: {metrics['f1']:.3f}
- Brier score: {metrics['brier_score']:.3f}

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

- Only {profile['record_count']} records are available, so model comparisons have high variance.
- The target is a publisher-supplied status label, not an independent outcome.
- Records span ATRS versions and publication years, creating structural and temporal shift.
- Public text can encode organisation-specific wording and process conventions.
- Repeated cross-validation reduces split sensitivity but does not create new independent evidence.
- Coefficients and feature effects are associations, not causal explanations.
- The serialized joblib artifact must be loaded only from a trusted checkout.

## Reproducibility

Run `python -m atrs_audit.modeling --input data/processed/atrs_audit.csv --output-root .` after installing `.[ml]`. Review `results/model_experiments.tsv`, `results/model_metrics.json`, and `results/oof_predictions.csv` before citing performance.
"""


def build_report(
    output_path: Path,
    results: Sequence[dict[str, Any]],
    selected: Experiment,
    profile: dict[str, Any],
    metrics: dict[str, Any],
    effects: Sequence[dict[str, Any]],
    generated_at: str,
) -> None:
    rows = []
    for result in results:
        auc = result.get("roc_auc_mean")
        auc_text = "crash" if auc is None else f"{float(auc):.3f} ± {float(result['roc_auc_std']):.3f}"
        bar_width = 0.0 if auc is None else max(0.0, min(100.0, 100 * float(auc)))
        rows.append(
            f"<tr><td>{html.escape(str(result['approach']))}</td><td>{auc_text}</td>"
            f"<td><div class='track'><div class='fill' style='width:{bar_width:.1f}%'></div></div></td>"
            f"<td>{html.escape(str(result['status']))}</td></tr>"
        )
    effect_rows = "".join(
        f"<tr><td>{html.escape(str(item['feature']))}</td><td>{float(item['coefficient']):+.3f}</td>"
        f"<td>{html.escape(str(item['direction']))}</td></tr>"
        for item in effects[:14]
    ) or "<tr><td colspan='3'>No linear coefficient view is available for the selected estimator.</td></tr>"
    confusion = metrics["confusion_matrix"]
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>ATRS Predictive Modelling Experiment</title><style>
:root{{--ink:#17202a;--muted:#607080;--paper:#f7f8f6;--card:#fff;--navy:#14344b;--teal:#087f75;--gold:#e3a83b;--line:#dce3e5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}
header{{background:linear-gradient(135deg,var(--navy),#1b6070);color:white;padding:58px 24px}}.wrap{{max-width:1080px;margin:auto}}h1{{font-size:clamp(2rem,5vw,3.7rem);line-height:1.05;margin:.15em 0}}
.lede{{max-width:780px;color:#e5f0f2;font-size:1.12rem}}main{{padding:34px 24px 70px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:15px;margin:24px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:21px;box-shadow:0 5px 18px rgba(20,52,75,.05)}}.kpi{{font-size:2.15rem;font-weight:800;color:var(--navy)}}.label{{color:var(--muted)}}
section{{margin:32px 0}}h2{{margin-top:0}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#eaf3f2}}
.track{{height:10px;min-width:150px;background:#e4ebed;border-radius:20px;overflow:hidden}}.fill{{height:100%;background:linear-gradient(90deg,var(--teal),#58b9ac)}}
.warning{{border-left:5px solid var(--gold);background:#fff7e7;padding:17px 19px}}.two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:17px}}small{{color:var(--muted)}}a{{color:#006b70}}
</style></head><body><header><div class='wrap'><small>PREDICTIVE MODELLING · PORTFOLIO EXPERIMENT</small><h1>Can public ATRS metadata distinguish production deployments?</h1>
<p class='lede'>Six reproducible classification experiments on the official GOV.UK Algorithmic Transparency Records snapshot, evaluated under one frozen repeated cross-validation design.</p></div></header><main class='wrap'>
<div class='warning'><strong>Not a deployment decision tool.</strong> This model predicts a publisher-supplied phase label from public disclosure patterns. It does not assess safety, fairness, effectiveness, legality, or governance quality.</div>
<div class='grid'><div class='card'><div class='kpi'>{profile['record_count']}</div><div class='label'>records</div></div><div class='card'><div class='kpi'>{metrics['roc_auc']:.3f}</div><div class='label'>aggregated OOF ROC-AUC</div></div><div class='card'><div class='kpi'>{metrics['balanced_accuracy']:.3f}</div><div class='label'>balanced accuracy</div></div><div class='card'><div class='kpi'>{metrics['brier_score']:.3f}</div><div class='label'>Brier score</div></div></div>
<section><div class='card'><h2>Experiment comparison</h2><p>Selected: <strong>{html.escape(selected.approach)}</strong>. Fold-level values show mean ± standard deviation across 25 validation folds.</p><div style='overflow:auto'><table><thead><tr><th>Approach</th><th>ROC-AUC</th><th>Visual</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div></section>
<section class='two'><div class='card'><h2>Data profile</h2><ul><li>{profile['production_records']} production / {profile['non_production_records']} non-production</li><li>Publication range: {profile['date_min']} to {profile['date_max']}</li><li>Primary metric: ROC-AUC</li><li>Validation: repeated 5-fold stratified CV</li><li>Random seed: {RANDOM_STATE}</li></ul></div>
<div class='card'><h2>Confusion matrix at 0.5</h2><table><tr><th></th><th>Predicted non-production</th><th>Predicted production</th></tr><tr><th>Actual non-production</th><td>{confusion['tn']}</td><td>{confusion['fp']}</td></tr><tr><th>Actual production</th><td>{confusion['fn']}</td><td>{confusion['tp']}</td></tr></table><small>Each probability is the mean of repeated out-of-fold predictions.</small></div></section>
<section><div class='card'><h2>Largest linear associations</h2><p>Shown from the fitted interpretable logistic model. Direction is associative, not causal.</p><div style='overflow:auto'><table><thead><tr><th>Feature</th><th>Coefficient</th><th>Direction</th></tr></thead><tbody>{effect_rows}</tbody></table></div></div></section>
<section class='warning'><strong>Limitations:</strong> {profile['record_count']} records are insufficient for a production model; the status label is not an independent outcome; version and time shifts remain; and public text may encode publisher-specific conventions. Read the <a href='../models/MODEL_CARD.md'>model card</a> before using these results.</section>
<small>Generated {html.escape(generated_at)} with scikit-learn {html.escape(sklearn.__version__)}.</small></main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def run(input_path: Path, output_root: Path) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    X, y = load_records(input_path)
    experiments = build_experiments()
    results = evaluate_experiments(experiments, X, y)
    selected = select_experiment(experiments, results)
    probabilities = out_of_fold_predictions(selected.estimator, X, y)
    metrics = prediction_metrics(y, probabilities)
    fitted = clone(selected.estimator).fit(X, y)

    # Coefficients come from the selected linear model, or from the explicit
    # interpretable challenger when a non-linear model wins.
    explanation_model = fitted
    explanation_model_id = selected.experiment_id
    method, effects = linear_feature_effects(explanation_model)
    if not effects:
        challenger = next(item for item in experiments if item.experiment_id == "exp_02_text_metadata_logistic")
        explanation_model = clone(challenger.estimator).fit(X, y)
        explanation_model_id = challenger.experiment_id
        method, effects = linear_feature_effects(explanation_model)
    for effect in effects:
        effect["model_id"] = explanation_model_id
        effect["interpretation"] = method

    dates = [row["date_published"] for row in X]
    profile = {
        "record_count": int(len(y)),
        "production_records": int(y.sum()),
        "non_production_records": int(len(y) - y.sum()),
        "positive_rate": round(float(y.mean()), 4),
        "date_min": min(dates),
        "date_max": max(dates),
        "duplicate_record_ids": int(len(X) - len({row["record_id"] for row in X})),
        "missing_descriptions": int(sum(not row["description"] for row in X)),
        "phase_counts": dict(Counter(row["phase"] for row in X)),
    }
    data_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "generated_at_utc": generated_at,
        "data_path": "data/processed/atrs_audit.csv",
        "data_sha256": data_hash,
        "target": TARGET_DEFINITION,
        "selected_experiment": selected.experiment_id,
        "selected_approach": selected.approach,
        "validation": "RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)",
        "selection_rule": "Highest mean ROC-AUC; prefer lower complexity within 0.01",
        "scikit_learn_version": sklearn.__version__,
        "python_random_state": RANDOM_STATE,
        "profile": profile,
        "oof_metrics": metrics,
    }

    results_dir = output_root / "results"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_log(results_dir / "model_experiments.tsv", results)
    (results_dir / "model_experiments.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "model_metrics.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        results_dir / "oof_predictions.csv",
        [
            {
                "record_id": row["record_id"],
                "title": row["title"],
                "actual_is_production": int(actual),
                "oof_probability_production": round(float(probability), 6),
            }
            for row, actual, probability in zip(X, y, probabilities)
        ],
    )
    write_csv(results_dir / "feature_effects.csv", effects)
    joblib.dump(
        {"model": fitted, "manifest": manifest},
        models_dir / "production_phase_classifier.joblib",
        compress=3,
    )
    (models_dir / "MODEL_CARD.md").write_text(
        build_model_card(selected, profile, metrics, generated_at), encoding="utf-8"
    )
    build_report(
        reports_dir / "model_evaluation.html",
        results,
        selected,
        profile,
        metrics,
        effects,
        generated_at,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/atrs_audit.csv"),
        help="Processed ATRS record-level CSV",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path.cwd(), help="Repository root"
    )
    args = parser.parse_args()
    manifest = run(args.input.resolve(), args.output_root.resolve())
    print(
        f"Selected {manifest['selected_approach']} with aggregated out-of-fold "
        f"ROC-AUC {manifest['oof_metrics']['roc_auc']:.3f}."
    )


if __name__ == "__main__":
    main()
