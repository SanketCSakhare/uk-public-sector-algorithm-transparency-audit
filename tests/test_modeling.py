import unittest
from pathlib import Path

import numpy as np

from atrs_audit.modeling import (
    build_experiments,
    load_records,
    prediction_metrics,
    select_experiment,
    select_metadata,
    select_text,
)


ROOT = Path(__file__).resolve().parents[1]


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_records(ROOT / "data/processed/atrs_audit.csv")

    def test_dataset_has_unique_binary_target(self):
        self.assertEqual(len(self.X), 142)
        self.assertEqual(set(self.y), {0, 1})
        self.assertEqual(int(self.y.sum()), 88)

    def test_feature_selectors_do_not_include_phase(self):
        text = select_text(self.X[:2])
        metadata = select_metadata(self.X[:2])
        self.assertEqual(len(text), 2)
        self.assertTrue(all("phase" not in key for row in metadata for key in row))

    def test_all_experiments_fit_and_predict_probabilities(self):
        for experiment in build_experiments():
            fitted = experiment.estimator.fit(self.X, self.y)
            probabilities = fitted.predict_proba(self.X[:3])[:, 1]
            self.assertEqual(probabilities.shape, (3,), experiment.experiment_id)
            self.assertTrue(np.all((0 <= probabilities) & (probabilities <= 1)))

    def test_prediction_metrics_include_confusion_matrix(self):
        probabilities = np.where(self.y == 1, 0.7, 0.3)
        metrics = prediction_metrics(self.y, probabilities)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 0)

    def test_selection_prefers_simpler_model_within_tolerance(self):
        experiments = build_experiments()
        results = [
            {"experiment_id": item.experiment_id, "complexity": item.complexity,
             "status": "evaluated", "roc_auc_mean": score}
            for item, score in zip(experiments, [0.5, 0.6818, 0.6900, 0.6675, 0.6245, 0.6338])
        ]
        selected = select_experiment(experiments, results)
        self.assertEqual(selected.experiment_id, "exp_01_metadata_logistic")


if __name__ == "__main__":
    unittest.main()
