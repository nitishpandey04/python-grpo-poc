"""Resource-limited Docker execution; CPython v3 by default, legacy OpenEnv opt-in."""
import json
import os
import re
import selectors
import subprocess
import time
import uuid


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip()


class Sandbox:
    def __init__(self, image=None, timeout=8, backend="cpython", memory="1g"):
        if backend not in {"cpython", "openenv"}:
            raise ValueError("Unknown execution backend")
        self.backend = backend
        self.image = image or ("python-grpo-cpython:v3" if backend == "cpython" else "python-grpo-sandbox:local")
        self.timeout = timeout
        self.memory = memory
        self.capabilities = []
        self.proc = None
        self.name = None
        self.buffer = b""

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Evaluation request timed out")
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Evaluation worker exited unexpectedly; inspect Docker state")
                self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def start(self):
        if self.proc is not None:
            return
        self.name = "python-grpo-" + uuid.uuid4().hex[:12]
        # Keep the container until explicit cleanup so a crash can be diagnosed.
        args = ["docker", "run", "-i", "--name", self.name,
                "--network=none", "--memory="+self.memory, "--cpus=1", "--pids-limit=64",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m", self.image]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.buffer = b""
        try:
            greeting = self._read(90)
            if greeting.get("ready") is not True or (self.backend == "cpython" and greeting.get("backend") not in {"cpython-v2","cpython-v3"}):
                raise RuntimeError("Unexpected evaluation worker greeting")
            self.capabilities=greeting.get("capabilities",[])
        except BaseException:
            self.close()
            raise

    def score(self, code, tests):
        if not tests:
            raise ValueError("Cannot score an empty test suite")
        return self._request({"code": extract_code(code), "tests": tests, "case_timeout": min(2, self.timeout / len(tests))})

    def score_stdio(self,code,cases,time_limit=2,memory_limit_mb=256):
        if not cases:raise ValueError('Empty standard-I/O suite')
        self.start()
        if 'stdio' not in self.capabilities:raise RuntimeError('Rebuild python-grpo-cpython:v3 for standard-I/O support')
        return self._request({'mode':'stdio','code':extract_code(code),'cases':cases,'time_limit':time_limit,'memory_limit_mb':memory_limit_mb},timeout=len(cases)*(float(time_limit)+0.5)+10)

    def score_plus(self, code, task_id):
        if self.backend != "cpython":
            raise ValueError("Expanded tests require the CPython backend")
        return self._request({"mode": "evalplus", "code": extract_code(code), "task_id": task_id}, timeout=90)

    def _request(self, request, timeout=None):
        self.start()
        tests = request.get("tests", request.get("cases", ["evalplus"]))
        start = time.monotonic()
        try:
            self.proc.stdin.write((json.dumps(request) + "\n").encode())
            self.proc.stdin.flush()
            response = self._read(timeout or self.timeout)
        except TimeoutError:
            self.close()
            return {"passed": 0, "total": len(tests), "status": "timeout", "errors": ["Execution exceeded time budget"], "seconds": time.monotonic() - start}
        except (RuntimeError, BrokenPipeError):
            state = subprocess.run(["docker", "inspect", "--format", "{{json .State}}", self.name], capture_output=True, text=True, timeout=10)
            details = json.loads(state.stdout) if state.returncode == 0 else {}
            self.close()
            # A generated program exceeding the enforced memory budget is a failed
            # rollout. Other infrastructure failures still abort rather than change rewards.
            if details.get("OOMKilled"):
                return {"passed": 0, "total": len(tests), "status": "memory_error", "errors": ["Generated code exceeded the 1 GiB sandbox memory budget"], "seconds": time.monotonic() - start}
            raise RuntimeError(f"Evaluation worker failed: {details}")
        if "infrastructure_error" in response:
            raise RuntimeError(response["infrastructure_error"])
        return response

    def close(self):
        if self.proc:
            subprocess.run(["docker", "rm", "-f", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            self.proc.communicate(timeout=20)
            self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


def program_score(sandbox,completion,tests,task_id=None):
    """Run one native Python test program, preserving setup and statement order."""
    program=tests if isinstance(tests,str) else '\n'.join(tests)
    if not program.strip():raise ValueError('Empty Python test program')
    score=sandbox.score(completion,[program])
    return {**score,'correct':score['total']==1 and score['passed']==1,'correctness_protocol':'python_test_program'}


def solution_score(sandbox,completion,tests,task_id=None):
    if isinstance(tests,dict):
        if tests.get('mode')!='stdio':raise ValueError('Unknown test execution mode')
        result=sandbox.score_stdio(completion,tests['cases'],tests['time_limit'],tests['memory_limit_mb'])
        return {**result,'correct':result['passed']==result['total'],'correctness_protocol':'codeforces_stdio'}
    return program_score(sandbox,completion,tests,task_id)
