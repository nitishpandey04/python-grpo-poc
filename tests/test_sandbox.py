"""Integration checks; explicitly invoked on a machine with Docker access."""
import unittest
from execution import Sandbox


class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sandbox = Sandbox(timeout=3)
        cls.sandbox.start()

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.close()

    def test_correct_and_incorrect_solutions(self):
        tests = ["assert add(1,2) == 3", "assert add(-1,1) == 0"]
        self.assertEqual(self.sandbox.score("def add(a,b):\n return a+b", tests)["passed"], 2)
        self.assertEqual(self.sandbox.score("def add(a,b):\n return a-b", tests)["passed"], 0)

    def test_errors_are_not_success(self):
        tests = ["assert f() == 1"]
        self.assertEqual(self.sandbox.score("def f(:", tests)["status"], "syntax_error")
        self.assertEqual(self.sandbox.score("def f():\n return 1/0", tests)["passed"], 0)

    def test_rejects_host_access(self):
        score = self.sandbox.score("import os\ndef f():\n return os.listdir('/')", ["assert f() == 1"])
        self.assertEqual(score["status"], "unsupported")

    def test_no_state_leakage(self):
        self.sandbox.score("def secret():\n return 1", ["assert secret() == 1"])
        self.assertEqual(self.sandbox.score("def f():\n return secret()", ["assert f() == 1"])["passed"], 0)

    def test_infinite_loop_is_bounded(self):
        score = self.sandbox.score("def f():\n while True:\n  pass", ["assert f() == 1"])
        self.assertIn(score["status"], ("timeout", "runtime_error", "ok"))
        self.assertEqual(score["passed"], 0)
        self.assertEqual(self.sandbox.score("def f():\n return 1", ["assert f() == 1"])["passed"], 1)

    def test_memory_limit_recovers_worker(self):
        score = self.sandbox.score("def f():\n return [0] * 500000000", ["assert f() == 1"])
        self.assertIn(score["status"], ("memory_error", "timeout"))
        self.assertEqual(score["passed"], 0)
        self.assertEqual(self.sandbox.score("def f():\n return 1", ["assert f() == 1"])["passed"], 1)


if __name__ == "__main__":
    unittest.main()
