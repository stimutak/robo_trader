"""Synthetic installation and process evidence; never invoke Gateway/Java."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from robo_trader.gateway_selection import (
    GatewaySelectionError,
    assert_running_gateway_matches,
    select_gateway,
)


def _install(base: Path, version: str) -> None:
    installation = base / f"IB Gateway {version}"
    (installation / "jars").mkdir(parents=True)
    (installation / "jars" / "gateway.jar").write_bytes(b"synthetic-only")
    (installation / "ibgateway.vmoptions").write_text("-Xmx768m")
    (installation / ".install4j").mkdir()
    (installation / ".install4j" / "i4jparams.conf").write_text(
        f'<application version="{version}.1p"/>'
    )


def _ibc(root: Path, version: str = "3.24.2") -> Path:
    path = root / "IBC release with spaces"
    (path / "scripts").mkdir(parents=True)
    (path / "version").write_text(version)
    (path / "IBC.jar").write_bytes(b"synthetic-only")
    for name in ("ibcstart.sh", "displaybannerandlaunch.sh"):
        script = path / "scripts" / name
        script.write_text("#!/bin/bash\nexit 0\n")
        script.chmod(0o755)
    return path


def test_new_install_wins_over_1037_and_pin_is_stable(tmp_path):
    base = tmp_path / "Applications"
    _install(base, "10.37")
    _install(base, "10.50")
    ibc = _ibc(tmp_path)
    env = {"ROBOTRADER_IBC_PATH": str(ibc)}
    selected = select_gateway(tmp_path, base, env)
    assert selected.version == "10.50"
    assert selected.build == "10.50.1p"
    _install(base, "10.51")
    env["GATEWAY_VERSION"] = selected.version
    assert select_gateway(tmp_path, base, env) == selected
    assert selected.launcher.name == "displaybannerandlaunch.sh"


@pytest.mark.parametrize("version", ["", "../10.51", "10.51.1p", "10.51\n", "1051"])
def test_invalid_pin_cannot_fall_back(tmp_path, version):
    _install(tmp_path, "10.51")
    with pytest.raises(GatewaySelectionError):
        select_gateway(tmp_path, tmp_path, {"GATEWAY_VERSION": version})


def test_latest_gateway_with_obsolete_ibc_fails_before_process_changes(tmp_path, monkeypatch):
    import scripts.gateway_manager as gm

    base = tmp_path / "Applications"
    _install(base, "10.51")
    ibc = _ibc(tmp_path, "3.23.0")
    monkeypatch.setenv("ROBOTRADER_IBC_PATH", str(ibc))
    monkeypatch.setenv("GATEWAY_VERSION", "10.51")
    monkeypatch.setattr(gm, "PLATFORM", "Darwin")
    monkeypatch.setattr(gm, "GATEWAY_BASE", base)
    monkeypatch.setattr(gm, "_ibc_safety_file_error", lambda: None)
    monkeypatch.setattr(gm, "stop_gateway", lambda: pytest.fail("must not stop"))
    monkeypatch.setattr(gm, "is_gateway_running", lambda: pytest.fail("must not accept process"))
    assert not gm.restart_gateway()
    assert not gm.start_gateway()


def test_missing_and_mismatched_build_metadata_rejected(tmp_path):
    base = tmp_path / "Applications"
    (base / "IB Gateway 10.51").mkdir(parents=True)
    ibc = _ibc(tmp_path)
    env = {"ROBOTRADER_IBC_PATH": str(ibc)}
    with pytest.raises(GatewaySelectionError, match="jars"):
        select_gateway(tmp_path, base, env)
    (base / "IB Gateway 10.51").rmdir()
    _install(base, "10.51")
    (base / "IB Gateway 10.51/.install4j/i4jparams.conf").write_text(
        '<application version="10.37.1p"/>'
    )
    with pytest.raises(GatewaySelectionError, match="mismatched"):
        select_gateway(tmp_path, base, env)


@pytest.mark.parametrize(
    "process_kind",
    ["old_gateway", "old_ibc", "manual", "unknown", "external_manual", "unknown_manual"],
)
def test_running_gateway_mismatch_never_passes_on_listener(tmp_path, monkeypatch, process_kind):
    base = tmp_path / "Applications"
    _install(base, "10.51")
    ibc = _ibc(tmp_path)
    selected = select_gateway(tmp_path, base, {"ROBOTRADER_IBC_PATH": str(ibc)})
    gateway = base / f"IB Gateway {'10.37' if process_kind == 'old_gateway' else '10.51'}"
    ibc_jar = tmp_path / "old/IBC.jar" if process_kind == "old_ibc" else ibc / "IBC.jar"
    command = f"java -cp {gateway}/jars/gateway.jar:{ibc_jar} ibcalpha.ibc.IbcGateway"
    if process_kind == "manual":
        command = f"{gateway}/IB Gateway.app/Contents/MacOS/launcher"
    if process_kind == "unknown":
        command = "java ibcalpha.ibc.IbcGateway"
    if process_kind == "external_manual":
        command = "/opt/IB Gateway 10.37/IB Gateway.app/Contents/MacOS/launcher"
    if process_kind == "unknown_manual":
        command = "unknown-launcher IB Gateway"
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, command)
    )
    with pytest.raises(GatewaySelectionError, match="running Gateway"):
        assert_running_gateway_matches(selected)


def test_gateway_manager_launches_exact_pair_without_vendor_defaults(tmp_path, monkeypatch):
    import scripts.gateway_manager as gm

    base = tmp_path / "Applications"
    _install(base, "10.51")
    ibc = _ibc(tmp_path)
    monkeypatch.setenv("GATEWAY_VERSION", "10.51")
    monkeypatch.setenv("ROBOTRADER_IBC_PATH", str(ibc))
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-child")
    monkeypatch.setattr(gm, "PLATFORM", "Darwin")
    monkeypatch.setattr(gm, "GATEWAY_BASE", base)
    monkeypatch.setattr(gm, "IBC_LOGS", tmp_path / "logs")
    monkeypatch.setattr(gm, "_ibc_safety_file_error", lambda: None)
    monkeypatch.setattr(gm, "assert_running_gateway_matches", lambda selection: None)
    monkeypatch.setattr(gm, "is_gateway_running", lambda: False)
    monkeypatch.setattr(gm, "is_api_port_listening", lambda port: True)
    launched = []
    monkeypatch.setattr(gm.subprocess, "Popen", lambda *a, **k: launched.append((a, k)))
    monkeypatch.setattr(gm.time, "sleep", lambda seconds: None)
    assert gm.start_gateway()
    args, kwargs = launched[0]
    assert args[0] == [str(ibc / "scripts/displaybannerandlaunch.sh")]
    assert kwargs["env"]["TWS_MAJOR_VRSN"] == "10.51"
    assert kwargs["env"]["IBC_PATH"] == str(ibc)
    assert kwargs["env"]["TRADING_MODE"] == "paper"
    assert kwargs["env"]["APP"] == "GATEWAY"
    for name in ("JAVA_PATH", "TWS_SETTINGS_PATH", "FIXUSERID", "FIXPASSWORD"):
        assert kwargs["env"][name] == ""
    assert "UNRELATED_SECRET" not in kwargs["env"]
    assert kwargs["start_new_session"] is True


def test_selection_cli_exports_safe_fields_from_synthetic_installations(tmp_path):
    import os
    import sys

    project_root = Path(__file__).resolve().parents[1]
    base = tmp_path / "Applications"
    _install(base, "10.51")
    ibc = _ibc(tmp_path)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "GATEWAY_VERSION": "10.51",
        "ROBOTRADER_IBC_PATH": str(ibc),
    }
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts/select_gateway.py")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.rstrip("\n").split("\t") == ["10.51", str(ibc), "3.24.2"]
