import json
import subprocess
import threading
import time
from pathlib import Path


class Monitor:
    def __init__(self, output, name, config):
        import trackio
        self.trackio = trackio
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.phase = "initializing"
        self.step = 0
        self.sample_count = 0
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.run = self.trackio.init(project="python-grpo", name=name, config=config, space_id=None, resume="allow" if config.get("resume") else "never", auto_log_gpu=False, auto_log_cpu=False)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def log(self, values):
        values = {k: v for k, v in values.items() if isinstance(v, (int, float))}
        with self.lock:
            self.run.log(values)
            with (self.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"time": time.time(), "phase": self.phase, "optimizer_step": self.step, **values}) + "\n")

    def sample(self, row):
        with self.lock:
            with (self.output / "rollouts.jsonl").open("a") as handle:
                handle.write(json.dumps({"time": time.time(), "optimizer_step": self.step, **row}) + "\n")
            self.sample_count += 1
            if self.step % 5 == 0 and self.sample_count % 4 == 0:
                self.run.log({"samples/solutions": self.trackio.Table(columns=["optimizer_step", "task_id", "completion", "reward", "status", "errors"], data=[[self.step, row["task_id"], row["completion"], row["reward"], row["status"], str(row["errors"])[:1500]]])})

    def _heartbeat(self):
        while not self.stop_event.is_set():
            status = {"phase": self.phase, "optimizer_step": self.step, "elapsed_seconds": round(time.monotonic() - self.started), "updated_at": time.time()}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
                if result.returncode == 0:
                    memory, utilization, temperature, power = map(float, result.stdout.strip().splitlines()[0].split(","))
                    self.log({"gpu/memory_mb": memory, "gpu/utilization_percent": utilization, "gpu/temperature_c": temperature, "gpu/power_w": power, "progress/optimizer_step": self.step, "progress/elapsed_seconds": status["elapsed_seconds"]})
            except (ValueError, OSError, subprocess.TimeoutExpired):
                pass
            tmp = self.output / "status.tmp"
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(self.output / "status.json")
            print("[heartbeat] " + json.dumps(status), flush=True)
            self.stop_event.wait(15)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        self.trackio.finish()
