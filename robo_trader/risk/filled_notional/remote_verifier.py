"""HTTPS client for an independently operated, pre-enrolled ledger authority.

This client does not implement the authority or prove its deployment independence.
It never enrolls a ledger, caches approval, follows redirects, or retries a write.
Runtime construction remains gated on provisioning and reviewing that service.
"""

import http.client
import json
import re
import secrets
import ssl
import threading
import time
from dataclasses import asdict
from urllib.parse import urlsplit

from .ledger import FilledNotionalUnavailable, MonotonicLedgerState

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_RESPONSE_BYTES = 8192
_CALL_TIMEOUT_SECONDS = 2.0


def _state_payload(state):
    if type(state) is not MonotonicLedgerState:
        raise ValueError("exact monotonic ledger state required")
    if type(state.ledger_id) is not str or not _LEDGER_ID.fullmatch(state.ledger_id):
        raise ValueError("invalid ledger identity")
    for count in (state.fill_count, state.conflict_count):
        if type(count) is not int or not 0 <= count <= 2**63 - 1:
            raise ValueError("invalid ledger counter")
    for head in (state.fill_head, state.conflict_head):
        if type(head) is not str or not _HASH.fullmatch(head):
            raise ValueError("invalid ledger head")
    return asdict(state)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response field")
        result[key] = value
    return result


class RemoteMonotonicVerifier:
    """Callable accepted by DailyFilledNotional; credentials supplied by its owner.

    The service must durably compare-and-advance state atomically before replying.
    Missing enrollment, rollback and forks must be rejected, including after its
    own restart. An unreachable authority makes the local ledger unavailable.
    """

    def __init__(self, *, endpoint: str, ledger_id: str, bearer_token: str):
        if type(endpoint) is not str or any(ord(c) <= 32 or ord(c) >= 127 for c in endpoint):
            raise ValueError("invalid monotonic authority endpoint")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("authority requires HTTPS without credentials, query or fragment")
        port = parsed.port
        if port is not None and port == 0:
            raise ValueError("invalid authority port")
        self._port = port if port is not None else 443
        if type(ledger_id) is not str or not _LEDGER_ID.fullmatch(ledger_id):
            raise ValueError("invalid enrolled ledger identity")
        if (
            type(bearer_token) is not str
            or not 1 <= len(bearer_token) <= 4096
            or any(ord(c) <= 32 or ord(c) >= 127 for c in bearer_token)
        ):
            raise ValueError("invalid authority credential")
        self._host = parsed.hostname
        self._path = parsed.path or "/"
        self._ledger_id = ledger_id
        self._token = bearer_token
        self._tls = ssl.create_default_context()
        self._tls.minimum_version = ssl.TLSVersion.TLSv1_2
        self._call_lock = threading.Lock()
        self._failed = False

    def __call__(self, candidate: MonotonicLedgerState) -> bool:
        """Bound the caller's wait, including otherwise unbounded system DNS.

        Only one transport worker may exist per client. A deadline latches the
        client unavailable permanently, even if that worker later completes. It
        may have advanced the remote head: recovery must reconcile that outcome,
        not retry blindly. A daemon worker closes its connection on completion;
        an OS resolver that never returns can strand at most this one worker.
        """
        deadline = time.monotonic() + _CALL_TIMEOUT_SECONDS
        if not self._call_lock.acquire(blocking=False):
            raise FilledNotionalUnavailable("remote monotonic verification already in progress")
        try:
            if self._failed:
                raise FilledNotionalUnavailable("remote monotonic verifier is latched unavailable")
            finished = threading.Event()
            outcome = []

            def run():
                try:
                    outcome.append(self._request(candidate))
                except Exception:
                    outcome.append(None)
                finally:
                    finished.set()

            threading.Thread(target=run, name="filled-notional-verifier", daemon=True).start()
            finished.wait(max(0.0, deadline - time.monotonic()))
            if not finished.is_set() or time.monotonic() >= deadline:
                self._failed = True
                raise FilledNotionalUnavailable("remote monotonic verification deadline exceeded")
            if not outcome or type(outcome[0]) is not bool:
                self._failed = True
                raise FilledNotionalUnavailable("remote monotonic verification unavailable")
            return outcome[0]
        except FilledNotionalUnavailable:
            self._failed = True
            raise
        except Exception:
            self._failed = True
            raise FilledNotionalUnavailable("remote monotonic verification unavailable") from None
        except BaseException:
            # Preserve interrupts but never permit a second worker afterward.
            self._failed = True
            raise
        finally:
            self._call_lock.release()

    def _request(self, candidate: MonotonicLedgerState) -> bool:
        connection = None
        try:
            state = _state_payload(candidate)
            if candidate.ledger_id != self._ledger_id:
                raise ValueError("candidate does not match enrolled ledger")
            request = {"protocol": 1, "nonce": secrets.token_hex(32), "state": state}
            connection = http.client.HTTPSConnection(
                self._host, self._port, timeout=2.0, context=self._tls
            )
            connection.request(
                "POST",
                self._path,
                body=json.dumps(request, separators=(",", ":")).encode("ascii"),
                headers={
                    "Authorization": "Bearer " + self._token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Cache-Control": "no-store",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError("authority did not return success")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ValueError("authority response exceeds limit")
            reply = json.loads(raw, object_pairs_hook=_unique_object)
            if type(reply) is not dict or set(reply) != {"protocol", "nonce", "state", "accepted"}:
                raise ValueError("invalid authority response fields")
            if (
                type(reply["protocol"]) is not int
                or reply["protocol"] != 1
                or reply["nonce"] != request["nonce"]
                or type(reply["accepted"]) is not bool
                or type(reply["state"]) is not dict
                or _state_payload(MonotonicLedgerState(**reply["state"])) != state
            ):
                raise ValueError("authority response is not bound to this request")
            return reply["accepted"]
        except Exception:
            # Neither credentials nor untrusted remote content belong in diagnostics.
            raise FilledNotionalUnavailable("remote monotonic verification unavailable") from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    raise FilledNotionalUnavailable(
                        "remote monotonic verification unavailable"
                    ) from None
