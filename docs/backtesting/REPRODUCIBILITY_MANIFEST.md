# Backtest Reproducibility Manifest

Status: offline contract implemented; engine orchestration is a later integration step.

The reproducibility manifest is the fail-closed identity for one backtest result. It does
not grant paper or live order authority, connect to a broker, or read production trading
state. A result is reviewable only when its bytes and every declared input still match the
manifest.

## Required provenance

Every finalized `BacktestRunManifest` contains:

- a schema version, run identifier, explicit creation timestamp, strategy identifier, and
  strategy version;
- one or more SHA-256 identities for each required input kind: `data`, `config`, `code`,
  and `model`;
- a SHA-256 package-lock identity plus the exact Python implementation/version, platform
  identifier, and sorted package versions used by the run;
- a root seed, generator identifier, derivation policy, and a derived seed for each named
  stochastic component;
- an IANA timezone, exchange calendar version, session policy, bar interval, bounded data
  window, corporate-action policy, price-adjustment policy, and missing-quote policy;
- a single split identifier and non-overlapping dataset partitions, including an explicit
  training partition and exactly one partition explicitly marked as the untouched holdout;
- commission, slippage, fill-timing, market-impact, partial-fill, and finalization policy
  identifiers;
- approval/error state; and
- a result artifact digest linked to the digest of all preceding inputs and assumptions.

A strategy that uses no serialized model must still provide an explicit model-set input,
for example a hash of `{"models": []}`. Omitting the model input is not equivalent to
declaring that no model was used.

## Deterministic hashing

`robo_trader.backtesting.provenance` provides the supported hashing boundaries:

- `digest_file()` opens a regular file with no-follow semantics, rejects symlink
  swaps, and rejects a file whose identity, size, or timestamps change while it
  is being hashed.
- `digest_file_set()` hashes relative paths and file digests under a declared root. File
  ordering does not change the result, but renaming a file does.
- `digest_dataframe()` covers index values/names, column labels, dtypes, cell values, and
  timezone-bearing timestamps.
- `digest_json()` hashes a typed canonical JSON representation for configuration and
  other structured inputs.

Canonical hashing rejects non-finite numbers, naive datetimes, non-string mapping keys,
arbitrary object stringification, duplicate DataFrame indexes/columns, and unsupported
types. Integers, binary floats, and decimals retain distinct encodings.

The manifest itself uses sorted compact JSON. `BacktestRunManifest.from_json()` accepts
only that canonical representation and exact schema keys. This makes byte-level storage,
review, and signatures unambiguous.

## Seed policy

Use `SeedPolicy.from_root(root_seed, component_ids)` to allocate a seed to every stochastic
component. The current derivation policy is `sha256-component-v1`; it derives unsigned
64-bit seeds from the root seed and component identifier. The manifest rejects a component
seed that does not match this derivation.

At minimum, a stochastic strategy run should name the execution simulator and strategy
components. Add identifiers for data augmentation, model fitting, parameter search, or any
other source of randomness. Components must use only their assigned seed and must not read
global random state.

## Finalizing and verifying a result

The caller first prepares immutable input objects, then finalizes the result in one call:

```python
manifest = BacktestRunManifest.create(
    run_id="candidate-42-holdout-1",
    created_at=run_started_at,
    strategy_id="mean-reversion",
    strategy_version="2.1.0",
    inputs=hashed_inputs,
    seed_policy=seed_policy,
    data_assumptions=data_assumptions,
    partitions=partitions,
    execution_assumptions=execution_assumptions,
    environment=environment,
    approval_eligible=True,
    recorded_errors=(),
    result_id="result-candidate-42",
    result_format_id="canonical-json-v1",
    result_payload=result_bytes,
)
```

`create()` hashes the exact result bytes and links them to a second digest computed from
all manifest inputs and assumptions plus the result identifier and format policy. The
returned dataclass and all nested contract values are frozen; sequences are normalized to
immutable tuples.

Before comparison, promotion, or readiness review, load the canonical manifest and verify
the exact stored result:

```python
manifest = BacktestRunManifest.from_json(manifest_text)
manifest.validate_for_approval(result_bytes)
```

Validation fails when result bytes change, any input or policy changes, a required hash is
missing, partitions overlap or escape the declared window, the untouched holdout is absent,
the package lock is not explicit, seed assignments mismatch, recorded errors exist, or the
run is not approval eligible.

## Input identity conventions

Identifiers are purpose labels, not substitutes for hashes. Recommended forms are:

- data: immutable dataset/version and vendor snapshot identifier;
- config: reviewed strategy-config schema version;
- code: Git tree or commit identifier plus the content hash of the selected source set;
- model: signed model identifier and file hash, or an explicit empty model-set hash;
- package lock: the exact resolved lock/requirements artifact used to construct the
  environment;
- partition: stable role and date-window identifier; and
- policy: a versioned implementation identifier whose semantics are documented in code.

Do not hash secrets into a manifest: even a one-way digest can enable guessing of a
low-entropy credential. Backtest manifests should reference only non-secret offline inputs.

## Integration boundary

This contract does not yet cause `BacktestEngine.run()` to emit a manifest automatically.
The integration PR must serialize results canonically, obtain actual code/config/model/data
hashes at the orchestration boundary, capture the selected environment, and pass the exact
engine/execution policies and seed assignments. It must not infer omitted values or mark a
recorded-error run approval eligible.
