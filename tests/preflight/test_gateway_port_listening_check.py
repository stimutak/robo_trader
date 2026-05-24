"""Unit tests for :class:`GatewayPortListeningCheck` (spec §9.1).

The check shells out to ``lsof`` and decodes the result. Every test
patches ``subprocess.run`` rather than executing real lsof so the suite
is hermetic across CI runners (macOS / linux lsof flag differences) and
fast.

The single most important invariant — enforced by ``mock_lsof`` in
``conftest.py`` and by the live-mode test below — is that the
implementation goes through ``subprocess.run(["lsof", ...])`` and NEVER
through ``socket.connect_ex``. The latter manufactures the very zombie
that breaks IBKR handshakes (CLAUDE.md 2025-12-06).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from robo_trader.preflight import CheckStatus, PreflightContext
from robo_trader.preflight.gateway_port_listening_check import GatewayPortListeningCheck

# Sample line shaped like real macOS lsof output. Format isn't parsed by
# the check (only stripped emptiness matters), but realistic fixtures
# make failures easier to debug at 3am.
SAMPLE_LISTEN_LINE = "java       1234 oliver   42u  IPv4 0xabc12345      0t0  TCP *:4002 (LISTEN)\n"


class TestGatewayPortListeningCheckPass:
    def test_returns_pass_when_lsof_finds_listener(self, mock_lsof, preflight_context):
        mock_lsof(returncode=0, stdout=SAMPLE_LISTEN_LINE)

        result = GatewayPortListeningCheck().run(preflight_context)

        assert result.status is CheckStatus.PASS
        assert result.name == "gateway_port_listening"
        assert "4002" in result.message
        assert result.details["port"] == 4002

    def test_pass_result_has_no_remediation(self, mock_lsof, preflight_context):
        # PASS results shouldn't carry remediation text — it's noise in
        # the JSON output and tooling may key off non-empty remediation
        # to flag "this needs operator attention."
        mock_lsof(returncode=0, stdout=SAMPLE_LISTEN_LINE)

        result = GatewayPortListeningCheck().run(preflight_context)

        assert result.remediation == ""


class TestGatewayPortListeningCheckBlock:
    def test_returns_block_when_lsof_returns_zero_but_empty_stdout(
        self, mock_lsof, preflight_context
    ):
        # Defensive: some lsof builds report rc=0 with an empty body
        # when given a filter that matches nothing. Treat as "nothing
        # listening" — same outcome as rc=1.
        mock_lsof(returncode=0, stdout="")

        result = GatewayPortListeningCheck().run(preflight_context)

        assert result.status is CheckStatus.BLOCK
        assert "4002" in result.message
        assert "not listening" in result.message

    def test_returns_block_when_lsof_returns_nonzero(self, mock_lsof, preflight_context):
        # rc=1 is the canonical "no matching socket" lsof exit.
        mock_lsof(returncode=1, stdout="")

        result = GatewayPortListeningCheck().run(preflight_context)

        assert result.status is CheckStatus.BLOCK
        assert "not listening" in result.message
        assert result.details["lsof_returncode"] == 1

    def test_block_remediation_names_start_gateway_path(self, mock_lsof, preflight_context):
        mock_lsof(returncode=1, stdout="")

        result = GatewayPortListeningCheck().run(preflight_context)

        # Remediation must point the operator at the canonical fix.
        assert "start_gateway" in result.remediation
        assert "2FA" in result.remediation

    def test_returns_block_when_lsof_times_out(self, mock_lsof, preflight_context):
        mock_lsof(side_effect=subprocess.TimeoutExpired(cmd="lsof", timeout=3))

        result = GatewayPortListeningCheck().run(preflight_context)

        assert result.status is CheckStatus.BLOCK
        assert "timed out" in result.message
        assert result.details["error"].startswith("lsof timeout")

    def test_returns_block_when_lsof_binary_missing(self, mock_lsof, preflight_context):
        mock_lsof(side_effect=FileNotFoundError("lsof"))

        result = GatewayPortListeningCheck().run(preflight_context)

        assert result.status is CheckStatus.BLOCK
        assert "lsof not installed" in result.message
        # Remediation must include the OS hint so a fresh-machine
        # operator doesn't have to think.
        assert "apt install" in result.remediation or "package manager" in result.remediation


class TestGatewayPortListeningCheckPortSelection:
    def test_uses_paper_port_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        captured: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout=SAMPLE_LISTEN_LINE, stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        # PreflightContext.for_test defaults target_port=4002 (paper).
        context = PreflightContext.for_test(tmp_path)
        GatewayPortListeningCheck().run(context)

        assert "-iTCP:4002" in captured["argv"]

    def test_uses_live_port_when_target_port_4001(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # Spec Q11.2: live mode (EXECUTION_MODE=live) wires the script
        # to construct the context with target_port=4001. The check
        # itself just reads context.target_port; verify it passes
        # through to the lsof argv.
        captured: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout=SAMPLE_LISTEN_LINE, stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        context = PreflightContext.for_test(tmp_path, target_port=4001)
        result = GatewayPortListeningCheck().run(context)

        assert "-iTCP:4001" in captured["argv"]
        # And the port surfaces in the human-readable message + details
        # so the operator can tell which port was probed.
        assert "4001" in result.message
        assert result.details["port"] == 4001


class TestGatewayPortListeningCheckSubprocessInvocation:
    def test_invokes_lsof_with_required_flags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # -nP avoids slow name resolution; -sTCP:LISTEN narrows to the
        # state we care about. If either flag is dropped, lsof either
        # hangs on DNS or returns mixed-state output and the rc/stdout
        # heuristic stops being meaningful. Lock the contract here.
        captured: dict[str, Any] = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout=SAMPLE_LISTEN_LINE, stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)

        GatewayPortListeningCheck().run(PreflightContext.for_test(tmp_path))

        argv = captured["argv"]
        assert argv[0] == "lsof"
        assert "-nP" in argv
        assert "-sTCP:LISTEN" in argv
        # Subprocess timeout must be set — defends against an lsof
        # hang independent of the runner's per-check budget.
        assert captured["kwargs"].get("timeout") == 3
        assert captured["kwargs"].get("capture_output") is True
        assert captured["kwargs"].get("text") is True


class TestGatewayPortListeningCheckUnexpectedErrors:
    def test_unexpected_oserror_propagates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # Per spec §5.4, checks should let truly unexpected errors
        # propagate; the runner wraps them into a synthesized BLOCK
        # with traceback. The check itself MUST NOT silently swallow
        # arbitrary exceptions and return a misleading PASS/BLOCK.
        class BoomError(RuntimeError):
            pass

        def fake_run(*args, **kwargs):
            raise BoomError("disk on fire")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(BoomError):
            GatewayPortListeningCheck().run(PreflightContext.for_test(tmp_path))


class TestGatewayPortListeningCheckMetadata:
    def test_implements_check_protocol_attrs(self):
        check = GatewayPortListeningCheck()
        assert check.name == "gateway_port_listening"
        assert check.description == "Gateway port listening"
        assert check.timeout_seconds == 3.0
