# Fractional historical-volume implementation

Status: version-2 contract, writer, readers and subprocess/data-fetcher integration implemented locally; full suite passed 3900 tests with 4 skips. Source-unit verification remains pending. This extends the current paper-readiness work;
it does not authorize orders or establish source units.

## Evidence and intended behavior

The worker now decodes historical volume as Decimal, but rejects fractions under
the version-1 integer canonical contract. The canonical SQLite table additionally
checks schema_version = 1 and declares volume INTEGER. An in-memory reproduction
shows SQLite stores `10.0000000000000001` as integer 10 in that column, even when
bound as text. TEXT affinity preserves the exact decimal string. Reproduction:
task-workspace `work/fractional-volume-storage-evidence.log`.

New broker batches must retain finite nonnegative decimal quantities through
JSON, canonical validation, storage, ownership fingerprints and readback. Raw
quantity remains distinct from shares; unknown Gateway units cannot produce
liquidity evidence. Existing version-1 records must remain unchanged and readable
with their original integer semantics.

## Implementation sequence

1. Add version-2 contract semantics: volume is Decimal and carries an explicit
   source-unit state, initially unknown. Keep a strict version-1 reader. Encode
   finite decimals as exact strings without float conversion or context-sensitive
   normalization. Reject booleans, nonfinite quantities and negative values.
2. Add a separate version-2 table with TEXT volume and the full canonical identity.
   Preserve the original table and all records. Do not rebuild, rewrite or delete
   operational data. Version-2 writes validate the complete batch and use existing
   atomic conflict-detection semantics; a conflict must not partially persist.
3. Update both async and synchronous readers. Choose a deterministic source series
   across supported versions, then bind its schema version in the follow-up query.
   Current read queries omit schema version because only version 1 exists; adding
   version 2 without revising them would permit mixed representation histories.
   Unknown versions and corrupt newest series must fail rather than fall back to
   older data that looks usable. Legacy presentation reads retain their limits.
4. Keep exact quantities in canonical objects and persisted JSON. Explicitly
   convert only the analytical DataFrame projection to a compatible numeric dtype;
   never use that projection as exact risk evidence. Ownership fingerprints must
   cover version, unit state and original exact quantity.
5. Change worker serialization only after the receiving contract and persistence
   path support version 2. Exercise raw protocol fields through the real decoder,
   worker handler, canonicalizer, SQLite writer and both readers. Preserve existing
   version-1 behavior and mixed-history deterministic selection.
6. Bind independently verified source units to the actual worker generation before
   producing dollar liquidity. A user-supplied environment label is not proof of
   the running Gateway setting. Unit verification remains a separate unresolved
   operational/integration requirement, not inferred from integral values.

## Acceptance evidence

- Exact round trips for integers beyond float precision, tiny fractional residue,
  ordinary fractional quantities, zero and very small positive values under low
  ambient Decimal precision.
- Rejection of malformed numeric text, unsupported versions, malformed units,
  partial or mixed-schema batches, conflicting event values and corrupt readback.
- No mutation of existing version-1 rows; repeat writes remain idempotent.
- Sync and async reader parity; explicit version identity prevents mixed series.
- Analytical consumers handle the projection without losing canonical ownership;
  source refresh/generation invalidation tests remain green.
- Full affected tests and independent review before enabling version-2 production
  writes. Broader repository verification after integration.

No operational database migration or launch is performed by this plan. Readiness
remains false until final risk admission, entry settlement and launch gates pass.


## IBC source-unit configuration (verified September 17)

The tracked template now sets `SendMarketDataInLotsForUSstocks=no`. This requests
that IBC clear the lots checkbox. It leaves `ReadOnlyApi=yes` unchanged. No actual
IBC config was changed and no Gateway restart was performed.

The [upstream IBC configuration reference](https://github.com/IbcAlpha/IBC/blob/master/resources/config.ini)
explains that accepting or deferring the size-display notification can set the
lots checkbox automatically; the explicit setting overrides that choice. Our
template already accepted that notification, but previously omitted the override.
The [IBC configuration action](https://github.com/IbcAlpha/IBC/blob/master/src/ibcalpha/ibc/ConfigureSendMarketDataInLotsForUSstocksTask.java)
logs whether the checkbox was already false or changed to false. It also logs an
error and returns if the checkbox is missing. Merely seeing the action start is
not evidence of success.

This is an operational path to requesting shares, not authenticated unit evidence.
Before liquidity admission, verify the installed IBC version supports this key,
that the intended config was used by the actual Gateway process, that the setting
was applied and persisted, and that evidence belongs to the current process/session.
Missing-checkbox errors, stale logs and desired configuration text must not mint
unit authority. Bind a verified current observation to the worker generation;
invalidate it on reconnect/settings changes. Keep unknown volume units rejected
by the liquidity producer until that integration exists.
