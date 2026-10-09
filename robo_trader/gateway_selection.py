"""Read-only selection of one Gateway/IBC pair for startup and recovery.

IBC's distributed gatewaystartmacos.sh assigns its own settings, overriding
exported variables. Launch the supplied banner/launcher directly instead;
never edit vendor scripts or rely on their default Gateway version.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class GatewaySelectionError(ValueError):
    """No validated installation pair is available."""


_GATEWAY_VERSION = re.compile(r"10\.[0-9]{2}")
_IBC_VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")


@dataclass(frozen=True)
class GatewaySelection:
    version: str
    ibc_path: Path
    gateway_base: Path
    ibc_version: str
    build: str

    @property
    def launcher(self) -> Path:
        return self.ibc_path / "scripts" / "displaybannerandlaunch.sh"


def select_gateway(
    project_root: Path,
    gateway_base: Path,
    environ: Mapping[str, str],
    *,
    version: str | None = None,
) -> GatewaySelection:
    """Validate before any lifecycle action; exported selection pins recovery."""
    configured = environ.get("GATEWAY_VERSION")
    if version is not None and configured is not None and version != configured:
        raise GatewaySelectionError("requested Gateway differs from pinned GATEWAY_VERSION")
    selected = version if version is not None else configured
    if selected is None:
        installed = (
            [
                item.name.removeprefix("IB Gateway ")
                for item in gateway_base.iterdir()
                if item.is_dir()
                and _GATEWAY_VERSION.fullmatch(item.name.removeprefix("IB Gateway "))
            ]
            if gateway_base.is_dir()
            else []
        )
        if not installed:
            raise GatewaySelectionError("no versioned IB Gateway installation found")
        selected = max(installed, key=lambda value: tuple(map(int, value.split("."))))
    if not _GATEWAY_VERSION.fullmatch(selected):
        raise GatewaySelectionError("GATEWAY_VERSION must be a major version such as 10.51")
    installation = gateway_base / f"IB Gateway {selected}"
    if not installation.is_dir():
        raise GatewaySelectionError(f"selected Gateway {selected} is not installed")
    if not (installation / "jars").is_dir() or not list((installation / "jars").glob("*.jar")):
        raise GatewaySelectionError("selected Gateway has no installed application jars")
    if not (installation / "ibgateway.vmoptions").is_file():
        raise GatewaySelectionError("selected Gateway has no ibgateway.vmoptions")
    try:
        metadata = (installation / ".install4j" / "i4jparams.conf").read_text()
    except OSError as exc:
        raise GatewaySelectionError("selected Gateway build metadata cannot be read") from exc
    builds = set(re.findall(r"(?<![0-9])10\.[0-9]{2}\.[0-9]+[a-z]?(?![0-9a-z])", metadata))
    if len(builds) != 1 or not next(iter(builds)).startswith(selected + "."):
        raise GatewaySelectionError("Gateway build metadata is missing, ambiguous or mismatched")
    override = environ.get("ROBOTRADER_IBC_PATH")
    ibc_path = Path(override) if override is not None else project_root / "IBCMacos-3"
    if not ibc_path.is_absolute() or any(c in str(ibc_path) for c in "\n\r\t"):
        raise GatewaySelectionError("ROBOTRADER_IBC_PATH must be an absolute single-line path")
    try:
        ibc_version = (ibc_path / "version").read_text().strip()
    except OSError as exc:
        raise GatewaySelectionError("selected IBC version file cannot be read") from exc
    match = _IBC_VERSION.fullmatch(ibc_version)
    if match is None:
        raise GatewaySelectionError("selected IBC version is malformed")
    minimum = (3, 24, 2) if int(selected.split(".")[1]) >= 48 else (3, 23, 0)
    if tuple(map(int, match.groups())) < minimum:
        raise GatewaySelectionError(
            f"Gateway {selected} requires IBC {'.'.join(map(str, minimum))} or newer"
        )
    for relative in ("IBC.jar", "scripts/displaybannerandlaunch.sh", "scripts/ibcstart.sh"):
        if not (ibc_path / relative).is_file():
            raise GatewaySelectionError(f"selected IBC is incomplete: missing {relative}")
    for relative in ("scripts/displaybannerandlaunch.sh", "scripts/ibcstart.sh"):
        if not (ibc_path / relative).stat().st_mode & 0o111:
            raise GatewaySelectionError(f"selected IBC script is not executable: {relative}")
    return GatewaySelection(selected, ibc_path, gateway_base, ibc_version, next(iter(builds)))


def assert_running_gateway_matches(selection: GatewaySelection) -> None:
    """Reject old or unidentifiable processes without starting or stopping them.

    IBC's JVM classpath contains the installed application jars and IBC.jar.
    This verifies the selected installation pair, not authentication/handshake.
    Full process commands may contain credentials and must never be reported.
    """
    result = subprocess.run(
        ["ps", "-axww", "-o", "command="], capture_output=True, text=True, timeout=5, check=False
    )
    if result.returncode:
        raise GatewaySelectionError("cannot identify existing Gateway processes")
    gateway_path = str(selection.gateway_base / f"IB Gateway {selection.version}") + "/"
    for command in result.stdout.splitlines():
        # Match every candidate recognized by gateway_manager.is_gateway_running.
        # A manual installation outside gateway_base must not evade validation.
        if "IbcGateway" not in command and "IB Gateway" not in command:
            continue
        if (
            "ibcalpha.ibc.IbcGateway" not in command
            or gateway_path not in command
            or str(selection.ibc_path / "IBC.jar") not in command
        ):
            raise GatewaySelectionError(
                "running Gateway does not match selected Gateway/IBC pair; "
                "a supervised connected-machine maintenance window is required"
            )
