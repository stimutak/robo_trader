# Walk-Forward and Final-Holdout Validation

Status: offline validation contract implemented; strategy-specific orchestration remains an
integration responsibility.

`robo_trader.backtesting.walk_forward` provides a fail-closed validation path for candidate
models and thresholds. It accepts only an offline, timezone-aware DataFrame and caller-owned
optimization/evaluation callbacks. It has no broker, runtime, database, credential, or order
authority.

## Window policy

`WalkForwardPlan` uses exact event counts for:

- fixed-size training data;
- an optional purge gap;
- fixed-size validation data used for parameters and thresholds;
- a second optional purge gap;
- fixed-size out-of-sample test data;
- a step size that must keep test windows non-overlapping;
- an optional embargo between consecutive test windows; and
- one fixed-size final holdout at the end of the dataset.

All train, validation, and test sizes must be positive. The validator never shrinks a window
to fit a short dataset. It rejects a plan with zero-sized test data, a step smaller than the
test plus embargo, no complete windows, overlapping tests, or any test event entering the
final holdout.

Training windows are rolling and fixed in length. A later window may use observations that
were historically available by that later decision time, but each optimizer receives only
that window's train and validation copies. It never receives test or holdout rows.

## Execution phases

For every window, in chronological order:

1. The validator copies the fixed training and validation slices.
2. It calls the optimizer with those slices and the window's derived optimizer seed.
3. The optimizer returns non-empty versioned model bytes, a hashed selected-config identity,
   and a finite validation score.
4. The validator copies the later test slice and calls the evaluator with a separate derived
   seed.
5. The evaluator returns exact ordered out-of-sample returns. Empty or non-finite results
   abort the entire run.
6. The validator records hashes and timestamp boundaries for all three slices, the model and
   selected configuration, both seeds, validation score, and OOS observations.

After all windows complete, the selection metric is computed from window test observations.
Only then is the already-selected model evaluated once on the final holdout. The holdout
callback cannot change which model was selected, and holdout returns are kept separate from
the aggregate walk-forward metrics.

Any callback exception, invalid return type, empty outcome, non-finite value, data/hash
mismatch, or incomplete window aborts with `WalkForwardValidationError`. There is no
skip-and-continue mode.

## Determinism

The plan owns one unsigned 64-bit root seed and generator identifier. The validator derives
distinct seeds for every window optimizer, every window evaluator, and the final holdout
evaluator through the manifest contract's `sha256-component-v1` policy. Callbacks must use
only their supplied seed and must not read global random state.

For the same canonical data, base config, plan, and deterministic callbacks,
`WalkForwardResult.to_bytes()` is byte-identical. A different seed, slice, model, selected
configuration, or observation changes the evidence digest.

## Out-of-sample metrics

The aggregate return sequence is constructed internally by concatenating only each window's
test returns in window order. `WalkForwardResult` rejects a manually constructed aggregate
that differs from those exact observations. Training scores, validation scores, and final
holdout returns cannot enter aggregate observation count, mean return, cumulative return,
volatility, or Sharpe ratio.

The final holdout has its own observation sequence and metrics. It remains visible for the
candidate readiness decision without being reused for tuning.

## Versioned evidence and manifest binding

The run requires exact `HashedInput` evidence for the full source DataFrame and base
configuration. It records, per window:

- train, validation, and test content hashes;
- model identifier, version, and bytes digest;
- selected-config identifier and digest;
- validation score and OOS returns; and
- deterministic optimizer/evaluator seeds.

`WalkForwardResult.model_evidence_digest()` covers every window model/config identity and
the selected window. Add that digest to the final `BacktestRunManifest` with:

```python
model_input = result.as_model_input("candidate-model-bundle-v1")
```

Serialize `result.to_bytes()` as the manifest's exact result payload. Then bind the completed
manifest:

```python
bound = result.bind_to_manifest(
    manifest,
    model_evidence_input_id="candidate-model-bundle-v1",
)
```

Binding verifies the exact result bytes; data, base-config, and model-bundle identities; seed
policy; timezone and data window; split and untouched-holdout identifiers; holdout content
hash; and holdout time boundaries. A mismatch fails rather than producing bound evidence.

The resulting `BoundWalkForwardEvidence` records both the full manifest digest and the exact
validation-result digest. It is evidence for review, not authorization to paper or live trade.

## Callback boundary

Optimizers and evaluators are strategy-specific, but their signatures and return contracts
are deliberately narrow:

```python
def optimizer(train, validation, seed) -> OptimizationOutcome:
    ...

def evaluator(model_outcome, oos_test, seed) -> EvaluationOutcome:
    ...
```

Callbacks must not close over the full dataset, future labels, the final holdout, or mutable
global state. The validator prevents accidental leakage through its arguments and evidence,
but Python cannot prevent a deliberately hostile callback from reading external globals.
Candidate-review code must therefore be trusted, versioned, content-hashed code included in
the manifest.
