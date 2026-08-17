import unittest

from atrs_audit.scoring import is_substantive, score_sections
from atrs_audit.pipeline import normalise_atrs_version


class ScoringTests(unittest.TestCase):
    def test_version_labels_are_normalised_without_losing_unknowns(self):
        self.assertEqual(normalise_atrs_version("v3"), "v3.0")
        self.assertEqual(normalise_atrs_version("3.0"), "v3.0")
        self.assertEqual(normalise_atrs_version("v4.0"), "v4.0")
        self.assertEqual(normalise_atrs_version("future"), "future")

    def test_missing_markers_are_not_substantive(self):
        for value in ("", "N/A", "not applicable", "TBC", "unknown"):
            self.assertFalse(is_substantive(value), value)

    def test_governance_fields_are_detected(self):
        sections = {
            "human-review": {
                "heading": "3.2 - Human review",
                "value": "A trained officer makes the final decision.",
            },
            "risks-and-mitigations": {
                "heading": "5.2 - Risks and mitigations",
                "value": "Bias drift is monitored monthly and triggers review.",
            },
            "model-performance": {
                "heading": "4.2.7 - Model performance",
                "value": "N/A",
            },
        }
        result = score_sections(sections)
        self.assertTrue(result["human_oversight"])
        self.assertTrue(result["risks_mitigations"])
        self.assertFalse(result["model_performance"])
        self.assertEqual(result["indicator_count"], 2)
        self.assertEqual(result["disclosure_score"], 25.0)


if __name__ == "__main__":
    unittest.main()
