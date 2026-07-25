import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAFETY = ROOT / "robo_trader" / "safety"


def test_safety_package_uses_standard_library_and_relative_imports_only():
    allowed_stdlib = set(getattr(sys, "stdlib_module_names", ()))
    for path in SAFETY.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in allowed_stdlib for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] in allowed_stdlib


def test_import_has_no_database_files_broker_modules_or_runtime_side_effects(tmp_path):
    script = f"""
import pathlib
import sqlite3
import sys

sys.path.insert(0, {str(ROOT)!r})
before = set(pathlib.Path.cwd().iterdir())
real_connect = sqlite3.connect
def forbidden_connect(*args, **kwargs):
    raise AssertionError("import attempted a database connection")
sqlite3.connect = forbidden_connect
import robo_trader.safety
after = set(pathlib.Path.cwd().iterdir())
assert before == after
for forbidden in (
    "ib_async",
    "robo_trader.runner_async",
    "robo_trader.execution",
    "robo_trader.order_manager",
    "robo_trader.stop_loss_monitor",
    "robo_trader.config",
):
    assert forbidden not in sys.modules, forbidden
sqlite3.connect = real_connect
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
    )


def test_active_runtime_startup_and_preflight_do_not_import_safety_package():
    targets = (
        ROOT / "robo_trader" / "runner_async.py",
        ROOT / "robo_trader" / "execution.py",
        ROOT / "robo_trader" / "order_manager.py",
        ROOT / "robo_trader" / "stop_loss_monitor.py",
        ROOT / "START_TRADER.sh",
    )
    targets += tuple((ROOT / "robo_trader" / "preflight").glob("*.py"))
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "robo_trader.safety" not in text
        assert "from .safety" not in text
        assert "from robo_trader import safety" not in text
