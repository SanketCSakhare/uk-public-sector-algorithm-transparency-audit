import unittest

from atrs_audit.parser import parse_sections


class SectionParserTests(unittest.TestCase):
    def test_extracts_heading_values_and_lists(self):
        body = """
        <h2 id="deployment">Deployment</h2>
        <h3 id="human-review">3.2 - Human review</h3>
        <p>A trained caseworker reviews every recommendation.</p>
        <ul><li>Escalation is available.</li><li>The final decision is human.</li></ul>
        <h3 id="appeals">3.5 - Appeals and review</h3><p>Users can request review.</p>
        """
        result = parse_sections(body)
        self.assertIn("human-review", result)
        self.assertIn("trained caseworker", result["human-review"]["value"])
        self.assertIn("final decision", result["human-review"]["value"])
        self.assertEqual(result["appeals"]["value"], "Users can request review.")

    def test_preserves_duplicate_ids(self):
        body = "<h3 id='data'>Data</h3><p>First value is meaningful.</p><h3 id='data'>Data</h3><p>Second value is meaningful.</p>"
        result = parse_sections(body)
        self.assertEqual(set(result), {"data", "data-2"})


if __name__ == "__main__":
    unittest.main()
