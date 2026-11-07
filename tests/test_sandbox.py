"""Integration checks: all candidate code executes only inside Docker."""
import unittest
from execution import Sandbox


class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sandbox = Sandbox(timeout=8)
        cls.sandbox.start()

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.close()

    def test_correct_and_partial_solutions(self):
        tests = ["assert add(1,2) == 3", "assert add(-1,1) == 0"]
        self.assertEqual(self.sandbox.score("def add(a,b):\n return a+b", tests)["passed"], 2)
        score = self.sandbox.score("def add(a,b):\n return 0", tests)
        self.assertEqual(score["passed"], 1)
        self.assertEqual(score["status"], "assertion_error")

    def test_valid_python_defaults_classes_and_prints(self):
        code = "class Counter:\n def __init__(self,n=1): self.n=n\n def value(self): return self.n\ndef f(n=2):\n print('output is captured')\n return Counter(n).value()"
        score = self.sandbox.score(code, ["assert f() == 2", "assert f(4) == 4"])
        self.assertEqual(score["passed"], 2)
        self.assertIn("output is captured",score["test_results"][0]["stdout"])

    def test_runtime_errors_during_tests_are_labelled(self):
        score = self.sandbox.score("def f():\n return 1/0", ["assert f() == 1"])
        self.assertEqual(score["status"], "runtime_error")
        self.assertEqual(score["test_results"][0]["exception_type"], "ZeroDivisionError")
        self.assertEqual(self.sandbox.score("def f(:", ["assert f() == 1"])["status"], "syntax_error")

    def test_empty_and_invalid_tests_fail_closed(self):
        with self.assertRaises(ValueError):
            self.sandbox.score("def f(): return 1", [])
        with self.assertRaises(RuntimeError):
            self.sandbox.score("def f(): return 1", ["assert ("])

    def test_container_isolation(self):
        code = "import os\ndef f():\n return os.getuid() != 0 and not os.listdir('/home') and 'HF_TOKEN' not in os.environ"
        self.assertEqual(self.sandbox.score(code,["assert f()"])["passed"],1)
        code = "from pathlib import Path\ndef f():\n Path('/app/cpython_worker.py').write_text('modified')"
        self.assertEqual(self.sandbox.score(code,["f()"])["status"],"runtime_error")

    def test_no_state_leakage_between_tests_or_requests(self):
        code = "values=[]\ndef f():\n values.append(1)\n return len(values)"
        self.assertEqual(self.sandbox.score(code,["assert f()==1","assert f()==1"])["passed"],2)
        self.sandbox.score("def secret(): return 1", ["assert secret()==1"])
        self.assertEqual(self.sandbox.score("def f(): return secret()", ["assert f()==1"])["status"],"runtime_error")

    def test_timeout_and_memory_failure_recover(self):
        score = self.sandbox.score("def f():\n while True: pass", ["assert f()==1"])
        self.assertEqual(score["status"], "timeout")
        score = self.sandbox.score("def f(): return [0]*500000000", ["assert f()==1"])
        self.assertEqual(score["status"],"memory_error")
        self.assertEqual(self.sandbox.score("def f(): return 1",["assert f()==1"])["passed"],1)

    def test_early_process_exit_is_not_success(self):
        code = "import os\ndef f(): os._exit(0)"
        score = self.sandbox.score(code,["assert f()==1"])
        self.assertEqual(score["status"],"runtime_error")
        self.assertEqual(score["passed"],0)

    def test_expanded_oracle_budget_is_explicit(self):
        result=self.sandbox.score_plus("def combinations_colors(l,n): return []",255)
        self.assertEqual(result["status"],"oracle_resource_limit",str(result))

    def test_expanded_tests_reject_constant_solution(self):
        correct = "def similar_elements(a,b): return tuple(set(a) & set(b))"
        good = self.sandbox.score_plus(correct,2)
        self.assertTrue(good.get("correct"),str(good))
        bad = self.sandbox.score_plus("def similar_elements(a,b): return (4,5)",2)
        self.assertFalse(bad.get("correct"),str(bad))
        self.assertGreater(good["suites"]["plus"]["total"],3)


if __name__ == "__main__":
    unittest.main()
