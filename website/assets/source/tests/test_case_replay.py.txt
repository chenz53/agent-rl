import json
import unittest

from nanoastra.case_replay import load_case, project_case
from nanoastra.clinical import read_record
from nanoastra.reward import compute_score


class CaseReplayTests(unittest.TestCase):
    def test_initial_view_does_not_contain_future_report_or_answer_key(self):
        view = project_case(load_case())
        text = json.dumps(view, ensure_ascii=False)
        for forbidden in ("腔内充盈缺损", "reference_diagnosis", "instructor_only", "objective_subtotal", "10:32"):
            self.assertNotIn(forbidden, text)
        self.assertIn("尚未完成", text)

    def test_stage_change_reveals_only_the_selected_report(self):
        case = load_case()
        initial = project_case(case, "initial")
        followup = project_case(case, "followup")
        records = lambda view: view["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
        self.assertIn("腔内充盈缺损", records(followup)["ctpa_report"])
        self.assertNotIn("腔内充盈缺损", records(initial)["ctpa_report"])
        self.assertIn("未完成", records(followup)["leg_ultrasound"])

    def test_projection_is_independent_and_private_fields_are_not_queryable(self):
        case = load_case()
        view = project_case(case)
        records = view["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
        records["risk_history"] = "edited"
        self.assertNotEqual(case["shared_records"]["risk_history"], "edited")
        for field in ("instructor_only", "label", "reference_diagnosis"):
            self.assertEqual(read_record(records, field, view["record_fields"]), {"status": "invalid_field"})

    def test_repeated_queries_do_not_advance_time(self):
        view = project_case(load_case())
        records = view["tools_kwargs"]["get_clinical_record"]["create_kwargs"]["record"]
        before = read_record(records, "ctpa_report", view["record_fields"])
        after = read_record(records, "ctpa_report", view["record_fields"])
        self.assertEqual(before, after)
        self.assertEqual(view["clock"], "D0 09:40")

    def test_invalid_stage_fails_and_old_scorer_cannot_score_new_task(self):
        with self.assertRaises(ValueError):
            project_case(load_case(), "tomorrow")
        with self.assertRaises(ValueError):
            compute_score("nanoastra_clinical_replay", "肺栓塞", "肺栓塞")


if __name__ == "__main__":
    unittest.main()
