import unittest
from audit_evaluation import summarize


class AuditTests(unittest.TestCase):
    def test_reference_and_coverage_filter_is_paired(self):
        score = {"correct":True,"status":"ok"}
        rows = [
            {"split":"test","task_id":1,"reference_valid":True,"models":{"baseline":score,"final":score},"legacy":{"baseline_correct":True,"final_correct":False},"expanded":{"eligible":True,"reason":"eligible","baseline":score,"final":score}},
            {"split":"test","task_id":2,"reference_valid":False,"models":{},"legacy":{"baseline_correct":True,"final_correct":True},"expanded":{"eligible":False,"reason":"invalid_original_reference"}},
            {"split":"test","task_id":3,"reference_valid":True,"models":{"baseline":score,"final":score},"legacy":{"baseline_correct":False,"final_correct":False},"expanded":{"eligible":False,"reason":"not_in_mbppplus"}},
        ]
        result=summarize(rows)
        self.assertEqual(result["test/original_cpython"]["problems"],2)
        self.assertEqual(result["test/expanded_compatible"]["problems"],1)
        self.assertEqual(result["test/original_cpython"]["historical_baseline_solved_on_same_tasks"],1)
        self.assertEqual(result["test/coverage"]["invalid_original_references"],[2])

    def test_no_compatible_tasks_does_not_report_perfect_accuracy(self):
        row={"split":"validation","task_id":1,"reference_valid":False,"models":{},"legacy":{},"expanded":{"eligible":False,"reason":"invalid_original_reference"}}
        result=summarize([row])
        self.assertIsNone(result["validation/expanded_compatible"]["baseline_correctness"])
        self.assertEqual(result["validation/expanded_compatible"]["problems"],0)


if __name__ == "__main__":
    unittest.main()
