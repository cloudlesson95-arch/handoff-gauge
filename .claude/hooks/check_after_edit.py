"""PostToolUse hook: after a Python file is edited, run ruff on it and the fast tests.

Exit code 2 sends the output back to the coding agent so it fixes the problem first.
"""

import json
import subprocess
import sys


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    path = (event.get("tool_input") or {}).get("file_path", "")
    if not path.endswith(".py"):
        return 0

    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", path], capture_output=True, text=True
    )
    if lint.returncode != 0:
        print(f"ruff found problems in {path}:\n{lint.stdout}", file=sys.stderr)
        return 2

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-m", "not slow", "--no-header"],
        capture_output=True,
        text=True,
    )
    # 5 = no tests collected (fresh project)
    if tests.returncode not in (0, 5):
        print(f"fast tests failed:\n{tests.stdout[-3000:]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
