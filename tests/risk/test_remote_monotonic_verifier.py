"""Remote approvals must be fresh, exact, and fail closed on transport errors."""

import json
import http.client
import socket
import ssl
import threading
import time
from dataclasses import replace

import pytest

from robo_trader.risk.filled_notional import FilledNotionalUnavailable, MonotonicLedgerState
from robo_trader.risk.filled_notional import remote_verifier as module

STATE = MonotonicLedgerState("a" * 32, 1, "b" * 64, 0, "0" * 64)


@pytest.fixture
def transport(monkeypatch):
    class Connection:
        status = 200
        transform = staticmethod(lambda body: body)
        calls = []
        closed = 0

        def __init__(self, host, port, *, timeout, context):
            assert host == "authority.example"
            assert port == 443
            assert context.check_hostname
            assert context.verify_mode == ssl.CERT_REQUIRED
            assert timeout == 2.0

        def request(self, method, path, *, body, headers):
            assert method == "POST"
            assert path == "/v1/verify"
            assert headers["Authorization"] == "Bearer test-secret"
            self.calls.append(json.loads(body))

        def getresponse(self):
            return self

        def read(self, limit):
            response = dict(self.calls[-1], accepted=True)
            body = self.transform(response)
            encoded = body if isinstance(body, bytes) else json.dumps(body).encode()
            return encoded[:limit]

        def close(self):
            type(self).closed += 1

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    return Connection


def verifier(**overrides):
    options = dict(
        endpoint="https://authority.example/v1/verify",
        ledger_id=STATE.ledger_id,
        bearer_token="test-secret",
    )
    options.update(overrides)
    return module.RemoteMonotonicVerifier(**options)


def test_exact_fresh_approval_and_no_cached_success(transport):
    check = verifier()
    assert check(STATE) is True
    assert check(STATE) is True
    assert transport.calls[0]["nonce"] != transport.calls[1]["nonce"]
    assert transport.closed == 2
    transport.status = 503
    with pytest.raises(FilledNotionalUnavailable):
        check(STATE)


@pytest.mark.parametrize(
    "change",
    [
        {"nonce": "old"},
        {"protocol": True},
        {"accepted": 1},
        {"extra": "field"},
        {"state": {}},
    ],
)
def test_wrong_or_ambiguous_response_fails(transport, change):
    transport.transform = staticmethod(lambda body: dict(body, **change))
    with pytest.raises(FilledNotionalUnavailable):
        verifier()(STATE)
    assert transport.closed == 1


def test_boolean_counter_cannot_match_integer_state(transport):
    def corrupt(body):
        body["state"]["fill_count"] = True
        return body

    transport.transform = staticmethod(corrupt)
    with pytest.raises(FilledNotionalUnavailable):
        verifier()(STATE)


@pytest.mark.parametrize(
    "body", [b"{}", b"not-json", b"x" * 8193, b'{"accepted":true,"accepted":false}']
)
def test_invalid_duplicate_or_oversized_response_fails(transport, body):
    transport.transform = staticmethod(lambda _: body)
    with pytest.raises(FilledNotionalUnavailable):
        verifier()(STATE)


def test_explicit_rejection_is_not_an_approval(transport):
    transport.transform = staticmethod(lambda body: dict(body, accepted=False))
    assert verifier()(STATE) is False


@pytest.mark.parametrize("status", [301, 302, 401, 403, 500])
def test_no_redirect_or_error_response_is_followed(transport, status):
    transport.status = status
    with pytest.raises(FilledNotionalUnavailable):
        verifier()(STATE)
    assert len(transport.calls) == 1
    assert transport.closed == 1


def test_transport_failure_is_sanitized_and_closed(transport):
    def fail(self):
        raise OSError("secret or private response")

    transport.getresponse = fail
    with pytest.raises(
        FilledNotionalUnavailable, match="remote monotonic verification unavailable"
    ):
        verifier()(STATE)
    assert transport.closed == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://authority.example/v1/verify",
        "https://user:pass@authority.example/v1/verify",
        "https://authority.example/v1/verify?token=secret",
        "https://authority.example/#fragment",
        "https://authority.example/\nheader",
        "https://authority.example:bad/verify",
        "https://authority.example:0/verify",
    ],
)
def test_invalid_endpoints_fail_before_network(transport, endpoint):
    with pytest.raises(ValueError):
        verifier(endpoint=endpoint)
    assert not transport.calls


def test_cleanup_failure_cannot_expose_private_exception(transport):
    def fail(self):
        raise OSError("private connection details")

    transport.close = fail
    with pytest.raises(FilledNotionalUnavailable) as error:
        verifier()(STATE)
    assert "private" not in str(error.value)


def test_real_ledger_uses_remote_checks_for_append_and_read(tmp_path, transport):
    from tests.risk.test_daily_filled_notional import _MONOTONIC_VERIFIERS, _fill, _service

    path = tmp_path / "accounting.db"
    # Test-only authority initializes fixture; production enrollment is separate.
    _service(path)
    authority = _MONOTONIC_VERIFIERS[str(path)]

    def compare_and_advance(body):
        return dict(body, accepted=authority(MonotonicLedgerState(**body["state"])))

    transport.transform = staticmethod(compare_and_advance)
    ledger = _service(path, monotonic_verifier=verifier(ledger_id=authority.state.ledger_id))
    result = ledger.record_fill(_fill("execution-1"))
    assert result.recorded
    assert authority.state.fill_count == 1
    assert ledger.record_fill(_fill("execution-1")).recorded is False
    transport.status = 503
    with pytest.raises(FilledNotionalUnavailable):
        ledger.record_fill(_fill("execution-2"))
    assert authority.state.fill_count == 1


def test_overall_deadline_latches_and_discards_late_approval(monkeypatch, transport):
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    original_close = transport.close

    def slow_response(self):
        entered.set()
        assert release.wait(5)
        return self

    def close(self):
        original_close(self)
        closed.set()

    transport.getresponse = slow_response
    transport.close = close
    monkeypatch.setattr(module, "_CALL_TIMEOUT_SECONDS", 0.05)
    check = verifier()
    start = time.monotonic()
    try:
        with pytest.raises(FilledNotionalUnavailable, match="deadline exceeded"):
            check(STATE)
        assert entered.is_set()
        assert time.monotonic() - start < 1
        with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
            check(STATE)
        assert len(transport.calls) == 1
    finally:
        release.set()
        assert closed.wait(2)
    with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
        check(STATE)
    assert len(transport.calls) == 1


def test_slow_progress_in_real_http_body_cannot_extend_deadline(monkeypatch, transport):
    client, server = socket.socketpair()
    client.settimeout(0.1)
    complete = threading.Event()
    closed = threading.Event()

    def getresponse(self):
        body = json.dumps(dict(self.calls[-1], accepted=True)).encode()

        def drip():
            try:
                server.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n"
                )
                for offset in range(0, len(body), 16):
                    server.sendall(body[offset : offset + 16])
                    time.sleep(0.01)
            finally:
                server.close()
                complete.set()

        threading.Thread(target=drip, daemon=True).start()
        self.response = http.client.HTTPResponse(client)
        self.response.begin()
        return self.response

    def close(self):
        self.response.close()
        client.close()
        closed.set()

    transport.getresponse = getresponse
    transport.close = close
    monkeypatch.setattr(module, "_CALL_TIMEOUT_SECONDS", 0.05)
    check = verifier()
    try:
        with pytest.raises(FilledNotionalUnavailable, match="deadline exceeded"):
            check(STATE)
        assert not complete.is_set()
        with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
            check(STATE)
    finally:
        assert complete.wait(3)
        assert closed.wait(3)


def test_thread_start_failure_latches_and_is_sanitized(monkeypatch, transport):
    starts = []

    def fail_start(self):
        starts.append(self)
        raise RuntimeError("private start failure")

    monkeypatch.setattr(module.threading.Thread, "start", fail_start)
    check = verifier()
    with pytest.raises(FilledNotionalUnavailable, match="verification unavailable"):
        check(STATE)
    with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
        check(STATE)
    assert len(starts) == 1
    assert not transport.calls


def test_interrupted_wait_cannot_start_another_worker(monkeypatch, transport):
    real_event = threading.Event
    entered, release, closed = real_event(), real_event(), real_event()
    created = []

    def response(self):
        entered.set()
        assert release.wait(3)
        return self

    def event():
        result = real_event()
        if not created:

            def interrupt(timeout=None):
                assert entered.wait(2)
                raise KeyboardInterrupt()

            result.wait = interrupt
        created.append(result)
        return result

    transport.getresponse = response
    transport.close = lambda self: closed.set()
    monkeypatch.setattr(module.threading, "Event", event)
    check = verifier()
    try:
        with pytest.raises(KeyboardInterrupt):
            check(STATE)
        with pytest.raises(FilledNotionalUnavailable, match="latched unavailable"):
            check(STATE)
        assert len(transport.calls) == 1
    finally:
        release.set()
        assert closed.wait(3)


@pytest.mark.parametrize(
    "state",
    [
        replace(STATE, ledger_id="c" * 32),
        replace(STATE, fill_count=True),
        replace(STATE, conflict_count=-1),
        replace(STATE, fill_head="bad"),
    ],
)
def test_wrong_enrollment_or_malformed_candidate_never_sent(transport, state):
    with pytest.raises(FilledNotionalUnavailable):
        verifier()(state)
    assert not transport.calls
