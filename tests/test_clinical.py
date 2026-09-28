import json
import unittest

from nanoastra.clinical import Case, DEMO_CASE, read_record, to_verl_row
from nanoastra.reward import compute_score


class ClinicalTests(unittest.TestCase):
    def test_answer_key_is_not_in_tool_or_prompt(self):
        case = Case("private-id", "presenting complaint", "positive", "negative", "SECRET_KEY")
        row = to_verl_row(case)
        self.assertNotIn("SECRET_KEY", json.dumps(row["prompt"]))
        self.assertNotIn("SECRET_KEY", json.dumps(row["extra_info"]))
        self.assertEqual(row["reward_model"]["ground_truth"], "SECRET_KEY")

    def test_unknown_fields_cannot_read_accidentally_supplied_private_data(self):
        record = {**DEMO_CASE.tool_record(), "label": "B"}
        for field in ("label", "ground_truth", "__dict__", "../label", ["exposure"], None):
            self.assertEqual(read_record(record, field), {"status": "invalid_field"})

    def test_missing_information_is_not_fabricated(self):
        self.assertEqual(read_record({}, "test_x"), {"status": "unavailable", "field": "test_x"})

    def test_records_are_independent(self):
        first = DEMO_CASE.tool_record()
        first["exposure"] = "changed"
        self.assertEqual(DEMO_CASE.tool_record()["exposure"], "positive")

    def test_terminal_reward_distinguishes_wrong_and_malformed(self):
        def score(text):
            return compute_score("nanoastra_synthetic", text, "B")
        self.assertEqual(score('<final>{"diagnosis":"B"}</final>')["score"], 1)
        wrong = score('<final>{"diagnosis":"A"}</final>')
        self.assertEqual((wrong["score"], wrong["format_valid"]), (0, 1))
        for text in (
            "B", '<final>{"diagnosis":["B"]}</final>', '<final>{"diagnosis":"C"}</final>',
            '<final>{"diagnosis":"B","extra":true}</final>',
            '<final>{"diagnosis":"B"}</final> extra',
            '<final>{"diagnosis":"A"}</final><final>{"diagnosis":"B"}</final>',
        ):
            self.assertEqual(score(text)["format_valid"], 0)
        self.assertEqual(score('tool response: negative\n<final>{"diagnosis":"B"}</final>')["score"], 1)

    def test_invalid_task_metadata_is_not_silently_scored(self):
        with self.assertRaises(ValueError):
            compute_score("real_clinical", "B", "B")
        with self.assertRaises(ValueError):
            compute_score("nanoastra_synthetic", "B", "C")


if __name__ == "__main__":
    unittest.main()
