"""Start persistent local processes; PID and logs are retained for inspection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def launch(kind, extra):
    env = dict(os.environ, HF_HOME=str(ROOT / ".cache/huggingface"), TRACKIO_DIR=str(ROOT / "runs/trackio"), TOKENIZERS_PARALLELISM="false")
    if kind == "dashboard":
        name = "dashboard"
        command = [sys.executable, "-u", "-c", "import trackio; trackio.show(project='python-grpo', open_browser=False, host='0.0.0.0', server_port=7860)"]
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", required=True)
        parsed, _ = parser.parse_known_args(extra)
        name = parsed.name
        command = [sys.executable, "-u", str(ROOT / "train.py"), *extra]
    log_dir = ROOT / "runs" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    pidfile = log_dir / "process.pid"
    if pidfile.exists():
        try:
            old_pid = int(pidfile.read_text())
            os.kill(old_pid, 0)
            cmdline = Path(f"/proc/{old_pid}/cmdline")
            if cmdline.exists() and cmdline.read_bytes():
                raise SystemExit(f"Process {old_pid} is already running; inspect {log_dir}")
        except ProcessLookupError:
            pass
    with (log_dir / "console.log").open("a") as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(str(proc.pid))
    print(f"Started {kind}: PID {proc.pid}; log: {log_dir / 'console.log'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["dashboard", "train"])
    args, extra = parser.parse_known_args()
    launch(args.kind, extra)
