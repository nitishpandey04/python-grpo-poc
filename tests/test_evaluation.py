import unittest
from evaluate import compare
from execution import extract_code


class EvaluationTests(unittest.TestCase):
    def test_comparison_pairs_by_id_not_order(self):
        before = [{"task_id": 1, "correct": False}, {"task_id": 2, "correct": True}]
        after = [{"task_id": 2, "correct": False}, {"task_id": 1, "correct": True}]
        result = compare(before, after)
        self.assertEqual(result["improvement_percentage_points"], 0)
        self.assertEqual((result["improved"], result["regressed"]), (1, 1))

    def test_comparison_rejects_different_tasks(self):
        with self.assertRaises(ValueError):
            compare([{"task_id": 1, "correct": True}], [{"task_id": 2, "correct": True}])

    def test_comparison_rejects_duplicate_tasks(self):
        rows=[{"task_id":1,"correct":True}]*2
        with self.assertRaises(ValueError):
            compare(rows,rows)

    def test_extracts_code_without_explanation(self):
        self.assertEqual(extract_code("Here is code:\n```python\ndef add(a,b):\n    return a+b\n```"), "def add(a,b):\n    return a+b")


if __name__ == "__main__":
    unittest.main()
