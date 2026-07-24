"""Regression tests for RoboTrader's import and distribution boundaries.

These tests intentionally use isolated child interpreters.  A passing import
must not depend on pytest's already-imported modules, the repository working
directory, or a real IBKR dependency.  Wheel construction also happens in a
temporary copy so setuptools cannot leave build artifacts in the source tree.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_python(
    script: str,
    *,
    cwd: Path,
    arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run a script without adding the repository or current directory to sys.path."""

    result = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(script), *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"isolated interpreter failed with exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def test_every_direct_ib_import_explicitly_imports_disconnect_guard() -> None:
    """Direct ``IB`` users cannot rely on a package-root monkey patch.

    Importing contract/data classes such as ``Stock`` does not create a
    connection and is deliberately outside this rule.  Any production module
    importing the stateful ``IB`` client must import its disconnect guard in
    that same module.
    """

    package_root = REPOSITORY_ROOT / "robo_trader"
    direct_ib_modules: set[str] = set()
    guarded_modules: set[str] = set()

    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        if "archived" in relative.parts:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_name = "robo_trader." + ".".join(relative.with_suffix("").parts)
        if module_name == "robo_trader.utils.ibkr_safe":
            # This is the guard implementation itself: it must import IB in
            # order to wrap it, and cannot meaningfully import itself.
            continue

        imports_direct_ib = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "ib_async"
            and any(alias.name == "IB" for alias in node.names)
            for node in ast.walk(tree)
        )
        if not imports_direct_ib:
            continue

        direct_ib_modules.add(module_name)
        imports_guard = any(
            (
                isinstance(node, ast.Import)
                and any(alias.name.endswith(".ibkr_safe") for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (
                    (node.module or "").endswith("ibkr_safe")
                    or (
                        (node.module or "").endswith("utils")
                        and any(alias.name == "ibkr_safe" for alias in node.names)
                    )
                )
            )
            for node in ast.walk(tree)
        )
        if imports_guard:
            guarded_modules.add(module_name)

    expected_direct_users = {
        "robo_trader.connection_manager",
        "robo_trader.clients.async_ibkr_client",
        "robo_trader.clients.ibkr_subprocess_worker",
        "robo_trader.utils.robust_connection",
        "robo_trader.utils.tws_health",
    }
    assert expected_direct_users <= direct_ib_modules
    assert direct_ib_modules == guarded_modules, (
        "modules importing ib_async.IB without an explicit ibkr_safe guard: "
        f"{sorted(direct_ib_modules - guarded_modules)}"
    )


def test_top_level_import_is_inert_and_does_not_load_ibkr(tmp_path: Path) -> None:
    """A cold ``import robo_trader`` must be metadata-only and side-effect free."""

    script = r"""
        import builtins
        import os
        from pathlib import Path
        import socket
        import subprocess
        import sys
        import threading

        repository_root = Path(sys.argv[1]).resolve()
        scratch = Path(sys.argv[2]).resolve()
        sys.path.insert(0, str(repository_root))
        sys.dont_write_bytecode = True

        forbidden_calls = []

        def forbidden(name):
            def fail(*args, **kwargs):
                forbidden_calls.append(name)
                raise AssertionError(f"import attempted {name}")
            return fail

        original_open = builtins.open

        def read_only_open(file, mode="r", *args, **kwargs):
            if any(character in mode for character in "wax+"):
                forbidden_calls.append(f"open:{mode}")
                raise AssertionError(f"import attempted write-mode open: {mode}")
            return original_open(file, mode, *args, **kwargs)

        builtins.open = read_only_open
        os.chdir = forbidden("os.chdir")
        subprocess.Popen = forbidden("subprocess.Popen")
        subprocess.run = forbidden("subprocess.run")
        subprocess.call = forbidden("subprocess.call")
        subprocess.check_call = forbidden("subprocess.check_call")
        subprocess.check_output = forbidden("subprocess.check_output")
        socket.socket = forbidden("socket.socket")
        socket.create_connection = forbidden("socket.create_connection")
        socket.getaddrinfo = forbidden("socket.getaddrinfo")
        threading.Thread.start = forbidden("thread.start")

        write_flags = (
            os.O_WRONLY
            | os.O_RDWR
            | os.O_APPEND
            | os.O_CREAT
            | os.O_TRUNC
        )
        mutation_events = {
            "os.chmod",
            "os.chown",
            "os.link",
            "os.mkdir",
            "os.remove",
            "os.rename",
            "os.replace",
            "os.rmdir",
            "os.symlink",
            "os.truncate",
            "subprocess.Popen",
        }

        def reject_mutating_audit_event(event, args):
            if event == "open":
                _path, mode, flags = args
                if (
                    (isinstance(mode, str) and any(c in mode for c in "wax+"))
                    or (isinstance(flags, int) and flags & write_flags)
                ):
                    raise AssertionError(f"import attempted mutating open: {args!r}")
            if event in mutation_events or event.startswith("socket."):
                raise AssertionError(f"import attempted {event}: {args!r}")

        sys.addaudithook(reject_mutating_audit_event)

        cwd_before = Path.cwd()
        environment_before = dict(os.environ)
        threads_before = tuple(thread.ident for thread in threading.enumerate())
        scratch_before = tuple(sorted(str(path.relative_to(scratch)) for path in scratch.rglob("*")))

        import robo_trader

        assert robo_trader.__version__ == "0.1.0"
        assert "ib_async" not in sys.modules
        assert "robo_trader.utils.ibkr_safe" not in sys.modules
        assert Path.cwd() == cwd_before
        assert dict(os.environ) == environment_before
        assert tuple(thread.ident for thread in threading.enumerate()) == threads_before
        assert tuple(sorted(str(path.relative_to(scratch)) for path in scratch.rglob("*"))) == scratch_before
        assert forbidden_calls == []
    """

    _run_isolated_python(
        script,
        cwd=tmp_path,
        arguments=(str(REPOSITORY_ROOT), str(tmp_path)),
    )


def test_explicit_ibkr_safe_import_installs_one_idempotent_guard(tmp_path: Path) -> None:
    """The opt-in safety import must patch a fake IB class once, without a broker."""

    script = r"""
        import os
        from pathlib import Path
        import sys
        import types

        repository_root = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(repository_root))
        sys.dont_write_bytecode = True

        calls = []

        class FakeIB:
            def isConnected(self):
                return True

            def disconnect(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "original-result"

        original_disconnect = FakeIB.disconnect
        fake_ib_async = types.ModuleType("ib_async")
        fake_ib_async.IB = FakeIB
        sys.modules["ib_async"] = fake_ib_async

        import robo_trader

        assert "robo_trader.utils.ibkr_safe" not in sys.modules

        from robo_trader.utils import ibkr_safe

        guarded_disconnect = FakeIB.disconnect
        assert guarded_disconnect is not original_disconnect
        assert guarded_disconnect.__name__ == "_patched_disconnect"

        ibkr_safe.patch_ib_disconnect()
        ibkr_safe.patch_ib_disconnect()
        assert FakeIB.disconnect is guarded_disconnect

        instance = FakeIB()
        os.environ.pop("IBKR_FORCE_DISCONNECT", None)
        assert instance.disconnect("not-forwarded") is False
        assert calls == []

        os.environ["IBKR_FORCE_DISCONNECT"] = "1"
        assert instance.disconnect("forwarded", reason="test") == "original-result"
        assert calls == [(("forwarded",), {"reason": "test"})]
    """

    _run_isolated_python(
        script,
        cwd=tmp_path,
        arguments=(str(REPOSITORY_ROOT),),
    )


def test_wheel_contains_supported_subpackages_and_excludes_runtime_artifacts(
    tmp_path: Path,
) -> None:
    """Build a copied tree and verify both wheel contents and isolated imports."""

    project_copy = tmp_path / "project"
    package_copy = project_copy / "robo_trader"
    project_copy.mkdir()
    shutil.copy2(REPOSITORY_ROOT / "pyproject.toml", project_copy / "pyproject.toml")
    shutil.copytree(
        REPOSITORY_ROOT / "robo_trader",
        package_copy,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        ignore_dangling_symlinks=True,
    )

    # Seed representative repository-only artifacts into the copied build
    # context.  A broad/unsafe package rule would accidentally ship these.
    (project_copy / "tests").mkdir()
    (project_copy / "tests" / "_package_boundary_sentinel.py").write_text(
        "SHOULD_NOT_SHIP = True\n",
        encoding="utf-8",
    )
    similarly_named_package = project_copy / "robo_trader_backup"
    similarly_named_package.mkdir()
    (similarly_named_package / "__init__.py").write_text(
        "SHOULD_NOT_SHIP = True\n",
        encoding="utf-8",
    )
    (project_copy / "config" / "ibc").mkdir(parents=True)
    (project_copy / "config" / "ibc" / "config.ini").write_text(
        "IbLoginId=do-not-package\n",
        encoding="utf-8",
    )
    (project_copy / ".env").write_text("SECRET=do-not-package\n", encoding="utf-8")
    (project_copy / "trading_data.db").write_bytes(b"not-a-real-database")
    (project_copy / "robo_trader.log").write_text("not-a-real-log\n", encoding="utf-8")

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    build_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--disable-pip-version-check",
            "--wheel-dir",
            str(wheelhouse),
            str(project_copy),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert build_result.returncode == 0, (
        f"wheel build failed with exit {build_result.returncode}\n"
        f"stdout:\n{build_result.stdout}\n"
        f"stderr:\n{build_result.stderr}"
    )

    wheels = list(wheelhouse.glob("robo_trader-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    wheel_import_root = tmp_path / "wheel-unpacked"
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        assert all(
            not Path(member).is_absolute() and ".." not in Path(member).parts for member in members
        )
        archive.extractall(wheel_import_root)

    # A regular package and a PEP 420 namespace package must both survive.
    assert "robo_trader/preflight/__init__.py" in members
    assert "robo_trader/preflight/result.py" in members
    assert "robo_trader/utils/pricing.py" in members
    assert "robo_trader/bug_detection/templates/bug_dashboard.html" in members

    assert not any(member.startswith("robo_trader/archived/") for member in members)
    assert not any(member.startswith("robo_trader_backup/") for member in members)
    assert not any(member.startswith("tests/") for member in members)
    assert not any(member.startswith("config/") for member in members)
    assert not any(Path(member).name == ".env" for member in members)
    assert not any(member.endswith((".db", ".sqlite", ".sqlite3", ".log")) for member in members)
    assert not any("config.ini" in member for member in members)

    import_script = r"""
        from pathlib import Path
        import sys

        wheel_root = Path(sys.argv[1]).resolve()
        sys.path.insert(0, str(wheel_root))

        import robo_trader
        from robo_trader.bug_detection.dashboard import BugDashboard
        import robo_trader.preflight.result as regular_module
        import robo_trader.utils.pricing as namespace_module

        class FakeBugAgent:
            bugs = []

            @staticmethod
            def generate_report():
                return {}

        response = BugDashboard(FakeBugAgent()).app.test_client().get("/")

        assert response.status_code == 200
        assert b"RoboTrader Bug Dashboard" in response.data
        assert "ib_async" not in sys.modules
        assert "robo_trader.utils.ibkr_safe" not in sys.modules
        assert str(wheel_root) in regular_module.__file__, regular_module.__file__
        assert str(wheel_root) in namespace_module.__file__, namespace_module.__file__
        assert str(wheel_root) in robo_trader.__file__, robo_trader.__file__
    """
    import_cwd = tmp_path / "wheel-import"
    import_cwd.mkdir()
    _run_isolated_python(
        import_script,
        cwd=import_cwd,
        arguments=(str(wheel_import_root),),
    )
