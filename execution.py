"""A persistent, resource-limited Docker worker using the real OpenEnv Coding API."""
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
    def __init__(self, image="python-grpo-sandbox:local", timeout=8):
        self.image, self.timeout = image, timeout
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
                    raise TimeoutError("OpenEnv execution timed out")
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("OpenEnv worker exited unexpectedly; inspect Docker / sandbox logs")
                self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def start(self):
        if self.proc is not None:
            return
        self.name = "python-grpo-" + uuid.uuid4().hex[:12]
        # Keep the container until explicit cleanup so a crash can be diagnosed.
        args = ["docker", "run", "-i", "--name", self.name,
                "--network=none", "--memory=1g", "--cpus=1", "--pids-limit=64",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m", self.image]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.buffer = b""
        try:
            if self._read(90) != {"ready": True}:
                raise RuntimeError("Unexpected OpenEnv worker greeting")
        except BaseException:
            self.close()
            raise

    def score(self, code, tests):
        self.start()
        start = time.monotonic()
        try:
            self.proc.stdin.write((json.dumps({"code": extract_code(code), "tests": tests}) + "\n").encode())
            self.proc.stdin.flush()
            response = self._read(self.timeout)
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
            raise RuntimeError(f"OpenEnv worker failed: {details}")
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
