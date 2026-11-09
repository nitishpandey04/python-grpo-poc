"""Integration checks: all candidate code executes only inside Docker."""
import unittest
from execution import Sandbox,program_score


class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sandbox = Sandbox(timeout=8)
        cls.sandbox.start()

    @classmethod
    def tearDownClass(cls):
        cls.sandbox.close()

    def test_standard_io_uses_real_python_stdin_and_normal_exit(self):
        tests=[{'input':'41\r\n','output':'42'}]
        code="import sys\nx=int(sys.stdin.buffer.read())\nprint(x+1)\nsys.exit(0)"
        self.assertTrue(self.sandbox.score_stdio(code,tests)['correct'])
        self.assertEqual(self.sandbox.score_stdio('int main() {}',tests)['status'],'syntax_error')

    def test_output_normalization_and_no_numeric_coercion(self):
        self.assertTrue(self.sandbox.score_stdio("print(' yes  No ')",[{'input':'','output':'YES\nNO'}])['correct'])
        self.assertFalse(self.sandbox.score_stdio("print('yesterday')",[{'input':'','output':'YESterday'}])['correct'])
        self.assertFalse(self.sandbox.score_stdio("print('1.0')",[{'input':'','output':'1'}])['correct'])

    def test_standard_io_is_fresh_and_fail_fast_is_explicit(self):
        code="from pathlib import Path\np=Path('state')\nx=int(p.read_text()) if p.exists() else 0\np.write_text(str(x+1))\nprint(x+1)"
        cases=[{'input':'','output':'1'}]*2
        self.assertTrue(self.sandbox.score_stdio(code,cases)['correct'])
        score=self.sandbox.score_stdio("print(0)",cases)
        self.assertFalse(score['correct']);self.assertEqual(score['checked'],1);self.assertEqual(score['total'],2)
        with self.assertRaises(ValueError):self.sandbox.score_stdio(code,[])

    def test_standard_io_limits_and_runtime_failure(self):
        score=self.sandbox.score_stdio('while True: pass',[{'input':'','output':'1'}],time_limit=0.2)
        self.assertEqual(score['status'],'timeout')
        score=self.sandbox.score_stdio('x=bytes(10**9)',[{'input':'','output':'1'}],memory_limit_mb=64)
        self.assertEqual(score['status'],'memory_error')
        self.assertFalse(self.sandbox.score_stdio('import os; os._exit(0)',[{'input':'','output':'1'}])['correct'])
        self.assertTrue(self.sandbox.score_stdio("import os; print(int(os.getuid()!=0 and 'HF_TOKEN' not in os.environ))",[{'input':'','output':'1'}])['correct'])

    def test_native_program_preserves_setup_and_explicit_failure(self):
        tests=['values=[1,2,3]','def check():',' for value in values:','  if f(value)!=value+1:','   raise ValueError("wrong result")','check()']
        self.assertTrue(program_score(self.sandbox,'def f(x): return x+1',tests)['correct'])
        self.assertFalse(program_score(self.sandbox,'def f(x): return x',tests)['correct'])
        self.assertTrue(program_score(self.sandbox,'def f(x): return x+1',['values=[1,2,3]','assert all(f(x)>x for x in values)'])['correct'])

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
