"""Docker-only worker. Never execute model code on the training host."""
import ast
import json
import sys
import time

from coding_env.models import CodeAction
from coding_env.server.python_codeact_env import PythonCodeActEnv

ALLOWED_IMPORTS = {"math", "collections", "itertools", "functools", "re", "string", "heapq", "bisect", "statistics"}
FORBIDDEN = {"eval", "exec", "compile", "open", "globals", "locals", "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "print", "safe_json_dumps", "format_exc"}


def validate(code):
    tree = ast.parse(code)
    if not tree.body or not any(isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError("Expected a Python function")
    for n in tree.body:
        if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            raise ValueError("Only function definitions and approved imports are supported")
    for n in ast.walk(tree):
        if isinstance(n, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef)):
            raise ValueError("Unsupported construct")
        if isinstance(n, ast.FunctionDef) and (n.decorator_list or n.args.defaults or n.args.kw_defaults and any(n.args.kw_defaults)):
            raise ValueError("Decorators and evaluated defaults are unsupported")
        if isinstance(n, ast.Name) and (n.id.startswith("__") or n.id in FORBIDDEN):
            raise ValueError("Unsupported name")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise ValueError("Private attributes are unsupported")
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
            raise ValueError("Attribute assignment is unsupported")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            roots = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [(n.module or "").split(".")[0]]
            if any(r not in ALLOWED_IMPORTS for r in roots) or getattr(n, "level", 0):
                raise ValueError("Unsupported import")
    return tree


def score(request):
    start = time.monotonic()
    code, tests = request["code"], request["tests"]
    result = {"passed": 0, "total": len(tests), "errors": [], "status": "ok"}
    try:
        validate(code)
    except (SyntaxError, ValueError) as exc:
        result.update(status="syntax_error" if isinstance(exc, SyntaxError) else "unsupported", errors=[str(exc)])
        return result
    # Separate executor per test: no cross-test state, no candidate-supplied test code.
    for test in tests:
        env = PythonCodeActEnv(additional_imports=sorted(ALLOWED_IMPORTS))
        env.reset()
        obs = env.step(CodeAction(code=code))
        if obs.exit_code:
            result["errors"].append(obs.stderr[-1200:])
            result["status"] = "runtime_error"
            continue
        obs = env.step(CodeAction(code=test))
        # Ignore OpenEnv's generic safety/conciseness reward. Only test success counts.
        if obs.exit_code == 0:
            result["passed"] += 1
        else:
            result["errors"].append(obs.stderr[-1200:])
    result["seconds"] = time.monotonic() - start
    return result


if __name__ == "__main__":
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            response = score(json.loads(line))
        except Exception as exc:
            response = {"infrastructure_error": repr(exc)}
        print(json.dumps(response), flush=True)
