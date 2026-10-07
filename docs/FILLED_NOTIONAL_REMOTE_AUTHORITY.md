# Independent filled-notional authority integration

Status: client implemented and locally tested; authority service, enrollment,
secret provisioning and runtime injection are **not implemented or deployed**.
Paper-entry readiness remains false. A successful client test does not establish
that an independent authority exists or retains its state durably.

## Client and wire contract

`robo_trader.risk.filled_notional.remote_verifier.RemoteMonotonicVerifier` is a
synchronous callable compatible with `DailyFilledNotional.monotonic_verifier`.
Construction requires an HTTPS endpoint, the explicitly enrolled 32-character
ledger ID, and a bearer credential obtained from an independent secret provider.
Never store that credential or the ledger HMAC key beside the database/anchor,
in the repository, or in logs. The bearer credential and HMAC key are distinct.
The client uses system CA validation, hostname checking and TLS 1.2 or newer.
It does not follow redirects, use proxy environment variables, enroll identities,
retry requests or cache approvals.

Each POST carries JSON with exactly these fields:

```json
{
  "protocol": 1,
  "nonce": "a fresh 64-character hexadecimal challenge",
  "state": {
    "ledger_id": "the enrolled 32-character hexadecimal identity",
    "fill_count": 1,
    "fill_head": "64 lowercase hexadecimal characters",
    "conflict_count": 0,
    "conflict_head": "64 lowercase hexadecimal characters"
  }
}
```

The authority returns HTTP 200 with exactly the same protocol, nonce and state,
plus `accepted`, an actual JSON boolean. A valid `false` is a rejection. Other
status codes, missing/extra/duplicate fields, changed identities/counts/heads,
nonmatching nonces, numeric substitutes for booleans, malformed JSON and bodies
over 8 KiB fail unavailable. Counters are nonnegative signed-64-bit integers.
No private response body or transport exception is included in client errors.

The two-second overall deadline bounds the calling ledger's wait, including DNS,
TLS, response headers, body and cleanup. A separate two-second socket timeout
does not provide that guarantee by itself. Only one transport worker may be
active per client; concurrent calls reject immediately. A transport error or
deadline permanently latches the client unavailable. A late reply never clears
that latch. The daemon worker closes its connection when the transport returns;
an OS resolver that never returns can strand at most one worker per client.
There is no automatic client replacement. Runtime integration must retain that
property, not reconstruct clients on every request or catch failures and retry.

## Required authority behavior before runtime use

The service must run and store accepted state outside the database/anchor failure
domain, with independent administration, authentication, access controls and a
reviewed recovery policy. Merely locating another file beside the database,
using a second path on the same restored volume, or supplying a callback that
returns true does not satisfy the requirement.

Enrollment is an explicit operator-reviewed operation binding a credential,
ledger identity and authenticated initial state. Verification must reject an
unknown identity; it must never implement trust-on-first-use. New ledger creation
and enrollment need a dedicated bootstrap workflow; the client cannot bypass a
constructor's first verification to manufacture an authoritative ledger.

The authority must atomically compare and durably persist state before approving:

1. Exact replay of its current accepted state is idempotent.
2. A transition advances exactly one of fill/conflict counts by one, preserves
   the unchanged chain head, and supplies a new advanced chain head.
3. Lower counts, skipped transitions, changed heads at unchanged counts, different
   identities and competing states at the same count are rejected.
4. Concurrent requests are serialized against durable state. Restart, lost state
   and restore cannot silently enroll or accept an earlier generation.

The local ledger authenticates its chains before requesting verification. The
remote authority prevents replay of an older valid database plus matching anchor;
it does not replace local chain validation or prove that executions are genuine.

A timeout can occur after remote persistence. Preserve local ledger/anchor
evidence and reconcile the remote head before constructing a replacement client.
An ambiguous outcome is not permission to reset the authority or retry a trading
action. No automatic local data repair or remote enrollment is provided here.

Before wiring startup, test the deployed service across concurrent advance/fork
requests, response loss after durable commit, service restart, coordinated local
database/anchor rollback and authority recovery. Review the exact enrollment,
key source and runtime lifecycle, then inject the same configured verifier into
the startup/replay/accounting path. These are outstanding launch gates.
