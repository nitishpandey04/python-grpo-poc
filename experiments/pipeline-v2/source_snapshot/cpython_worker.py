"""Docker-only CPython evaluator v2. Candidate code is never run on the host."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

VERSION = "cpython-v2"


class BoundedOutput(io.StringIO):
    def write(self, text):
        size = len(text)
        remaining = max(0, 4096 - self.tell())
        if remaining:
            super().write(text[:remaining])
        return size


def limits(cpu_seconds=3, memory=512 * 1024 * 1024):
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def case(request):
    """One fresh CPython process, one test, with precise outcome classification."""
    limits()
    sink = sys.stdout
    capture = BoundedOutput()
    result = {"status": "ok", "exception_type": None, "message": ""}
    namespace = {"__name__": "__candidate__"}
    try:
        candidate = compile(request["code"], "<candidate>", "exec")
    except SyntaxError as exc:
        return {"status": "syntax_error", "exception_type": "SyntaxError", "message": str(exc)}
    test = compile(request["test"], "<test>", "exec")
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            exec(candidate, namespace)
            exec(test, namespace)
    except AssertionError as exc:
        result.update(status="assertion_error", exception_type="AssertionError", message=str(exc))
    except MemoryError:
        result.update(status="memory_error", exception_type="MemoryError", message="Exceeded per-process address-space budget")
    except BaseException as exc:
        result.update(status="runtime_error", exception_type=type(exc).__name__, message=str(exc)[:1000])
    result["stdout"] = capture.getvalue()
    return result


def subprocess_case(request, mode="case", timeout=2):
    proc = subprocess.Popen([sys.executable, "-P", "-s", __file__, "--" + mode],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True, env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MBPP_OVERRIDE_PATH": "/app/MbppPlus-v0.2.0.jsonl.gz", "EVALPLUS_MAX_MEMORY_BYTES": "2147483648"})
    try:
        stdout, stderr = proc.communicate(json.dumps(request), timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return {"status": "timeout", "exception_type": "TimeoutError", "message": f"Exceeded {timeout}s wall-time budget"}
    # No response (including os._exit(0)) must never count as passing.
    if proc.returncode:
        status = "memory_error" if proc.returncode == -signal.SIGKILL else "runtime_error"
        return {"status": status, "exception_type": "ProcessExit", "message": f"Child exited with code {proc.returncode}: {stderr[-1000:]}"}
    try:
        response = json.loads(stdout)
        if not isinstance(response, dict) or "status" not in response:
            raise ValueError("Missing status")
        return response
    except (ValueError, TypeError):
        return {"status": "runtime_error", "exception_type": "ProtocolError", "message": "Child did not return a complete evaluation result"}


def score(request):
    code, tests = request["code"], request["tests"]
    if not tests or not all(isinstance(t, str) and t.strip() for t in tests):
        raise ValueError("Evaluation requires at least one nonempty trusted test")
    # Invalid author-supplied tests are an infrastructure/data error, not a model failure.
    for test in tests:
        compile(test, "<trusted-test>", "exec")
    if len(code.encode()) > 65536:
        return {"passed": 0, "total": len(tests), "status": "runtime_error", "errors": ["Candidate exceeds 64 KiB source budget"], "test_results": [], "evaluator": VERSION}
    start = time.monotonic()
    outcomes = [subprocess_case({"code": code, "test": test}, timeout=request.get("case_timeout", 2)) for test in tests]
    passed = sum(row["status"] == "ok" for row in outcomes)
    failures = [row for row in outcomes if row["status"] != "ok"]
    priority = {"memory_error": 0, "timeout": 1, "syntax_error": 2, "runtime_error": 3, "assertion_error": 4}
    status = min(failures, key=lambda row: priority[row["status"]])["status"] if failures else "ok"
    return {"passed": passed, "total": len(tests), "status": status,
            "errors": [f"{row['exception_type']}: {row['message']}" for row in failures],
            "test_results": outcomes, "seconds": time.monotonic() - start, "evaluator": VERSION}


def plus_case(request):
    """Use EvalPlus's own input deserialization, special oracles, and checker."""
    import copy
    import numpy as np
    from evalplus.data import get_mbpp_plus
    from evalplus.eval import untrusted_check
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec

    limits(cpu_seconds=65, memory=2 * 1024 * 1024 * 1024)
    problem = get_mbpp_plus()[f"Mbpp/{request['task_id']}"]
    canonical = problem["prompt"] + problem["canonical_solution"]
    result = {"status": "ok", "evaluator": "evalplus-0.3.1-mbppplus-v0.2.0", "suites": {}}
    nonnull = problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
    for name, inputs in [("base", problem["base_input"]), ("plus", problem["plus_input"])]:
        if not inputs:
            raise ValueError("Empty EvalPlus input suite")
        # Canonical code and generated code both execute only inside Docker.
        expected, ref_times = trusted_exec(canonical, inputs, problem["entry_point"], record_time=True, output_not_none=nonnull)
        status, details = untrusted_check("mbpp", request["code"], copy.deepcopy(inputs),
                                         problem["entry_point"], expected, problem["atol"], ref_times,
                                         fast_check=True, min_time_limit=0.2, gt_time_limit_factor=4.0)
        result["suites"][name] = {"status": status, "correct": status == "pass", "total": len(inputs), "checked": len(details)}
    result["correct"] = all(row["correct"] for row in result["suites"].values())
    return result


def main():
    print(json.dumps({"ready": True, "backend": VERSION}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("mode") == "evalplus":
                response = subprocess_case(request, mode="evalplus", timeout=75)
            else:
                response = score(request)
        except BaseException as exc:
            response = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        request = json.load(sys.stdin)
        if sys.argv[1] == "--case":
            result = case(request)
        else:
            try:
                result = plus_case(request)
            except MemoryError:
                result = {"status": "oracle_resource_limit", "message": "Canonical oracle exceeded the expanded-suite address-space budget"}
            except BaseException as exc:
                result = {"status": "oracle_error", "message": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result), flush=True)
    else:
        main()
