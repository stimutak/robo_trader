"""Leakage, boundary, failure, and determinism tests for walk-forward validation."""

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from robo_trader.backtesting.provenance import (
    HashedInput,
    digest_dataframe,
    hashed_json_input,
)
from robo_trader.backtesting.walk_forward import (
    EvaluationOutcome,
    OptimizationOutcome,
    WalkForwardPlan,
    WalkForwardValidationError,
    WalkForwardValidator,
)


def _data(size: int = 60) -> pd.DataFrame:
    index = pd.date_range("2020-01-02 16:00", periods=size, freq="1D", tz="America/New_York")
    return pd.DataFrame(
        {
            "feature": np.arange(size, dtype=float),
            "return": np.arange(size, dtype=float) / 10_000,
        },
        index=index,
    )


def _plan(**overrides) -> WalkForwardPlan:
    values = {
        "split_id": "rolling-v1",
        "holdout_partition_id": "final-holdout",
        "train_size": 10,
        "validation_size": 5,
        "test_size": 4,
        "holdout_size": 8,
        "step_size": 4,
        "purge_size": 1,
        "embargo_size": 0,
        "root_seed": 42,
        "selection_metric": "mean_return",
    }
    values.update(overrides)
    return WalkForwardPlan(**values)


def _inputs(data: pd.DataFrame) -> Tuple[HashedInput, HashedInput]:
    return (
        HashedInput("data", "prices-v1", digest_dataframe(data)),
        hashed_json_input("config", "base-config-v1", {"lookback": 10}),
    )


def _callbacks(trace: Dict[str, List]):
    def optimizer(train: pd.DataFrame, validation: pd.DataFrame, seed: int):
        trace.setdefault("optimizer", []).append(
            (train.index.copy(), validation.index.copy(), seed)
        )
        selected_config = hashed_json_input(
            "config",
            f"selected-{int(train.iloc[0]['feature'])}",
            {"threshold": float(validation["feature"].mean())},
        )
        return OptimizationOutcome(
            model_id=f"model-{int(train.iloc[0]['feature'])}",
            model_version="1.0.0",
            model_bytes=f"model:{seed}:{train.index[-1].isoformat()}".encode(),
            selected_config=selected_config,
            validation_score=float(validation["return"].mean()),
        )

    def evaluator(outcome: OptimizationOutcome, test: pd.DataFrame, seed: int):
        trace.setdefault("evaluator", []).append((outcome.model_id, test.index.copy(), seed))
        return EvaluationOutcome(tuple(test["return"].tolist()))

    return optimizer, evaluator


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("train_size", 0, "train_size"),
        ("validation_size", 0, "validation_size"),
        ("test_size", 0, "test_size"),
        ("holdout_size", 0, "holdout_size"),
        ("step_size", 0, "step_size"),
        ("purge_size", -1, "purge_size"),
        ("embargo_size", -1, "embargo_size"),
    ],
)
def test_plan_rejects_zero_or_negative_window_sizes(field, value, match) -> None:
    with pytest.raises(ValueError, match=match):
        _plan(**{field: value})


def test_plan_rejects_shrinking_or_overlapping_test_schedule() -> None:
    with pytest.raises(ValueError, match=r"test_size \+ embargo_size"):
        _plan(test_size=4, embargo_size=2, step_size=5)


def test_fixed_windows_are_ordered_purged_and_exclude_holdout() -> None:
    plan = _plan(purge_size=2, embargo_size=1, step_size=5)
    windows = WalkForwardValidator(plan).build_windows(60)
    holdout_start = 60 - plan.holdout_size

    assert windows
    for window in windows:
        assert window.train_end - window.train_start == plan.train_size
        assert window.validation_end - window.validation_start == plan.validation_size
        assert window.test_end - window.test_start == plan.test_size
        assert window.validation_start - window.train_end == plan.purge_size
        assert window.test_start - window.validation_end == plan.purge_size
        assert window.test_end + plan.purge_size <= holdout_start
    for previous, current in zip(windows, windows[1:]):
        assert previous.test_end + plan.embargo_size <= current.test_start


def test_each_fit_sees_only_past_train_and_validation_not_test_or_holdout() -> None:
    data = _data()
    plan = _plan()
    trace: Dict[str, List] = {}
    optimizer, evaluator = _callbacks(trace)
    source, config = _inputs(data)

    result = WalkForwardValidator(plan).run(
        data,
        source_data=source,
        base_config=config,
        optimizer=optimizer,
        evaluator=evaluator,
    )

    holdout_start = data.index[-plan.holdout_size]
    assert len(trace["optimizer"]) == len(result.windows)
    for evidence, (train_index, validation_index, _seed) in zip(result.windows, trace["optimizer"]):
        test_index = data.index[evidence.boundary.test_start : evidence.boundary.test_end]
        assert train_index[-1] < validation_index[0]
        assert validation_index[-1] < test_index[0]
        assert validation_index[-1] < holdout_start
        assert set(train_index).isdisjoint(test_index)
        assert set(validation_index).isdisjoint(test_index)
    assert all(index[-1] < holdout_start for _, index, _ in trace["evaluator"][:-1])
    assert trace["evaluator"][-1][1].equals(data.index[-plan.holdout_size :])


def test_holdout_is_evaluated_only_after_model_selection_and_cannot_change_it() -> None:
    data = _data()
    plan = _plan()
    source, config = _inputs(data)
    events = []

    def optimizer(train, validation, seed):
        events.append(("optimize", train.index[-1]))
        model_number = int(train.iloc[0]["feature"])
        return OptimizationOutcome(
            model_id=f"model-{model_number}",
            model_version="1",
            model_bytes=f"{model_number}:{seed}".encode(),
            selected_config=hashed_json_input(
                "config", f"selected-{model_number}", {"number": model_number}
            ),
            validation_score=0,
        )

    def evaluator(outcome, test, seed):
        is_holdout = test.index[0] == data.index[-plan.holdout_size]
        events.append(("holdout" if is_holdout else "test", outcome.model_id))
        if is_holdout:
            return EvaluationOutcome((999.0,))
        model_number = int(outcome.model_id.split("-")[-1])
        return EvaluationOutcome((float(model_number),))

    result = WalkForwardValidator(plan).run(
        data,
        source_data=source,
        base_config=config,
        optimizer=optimizer,
        evaluator=evaluator,
    )

    selected = max(result.windows, key=lambda window: window.oos_metrics["mean_return"])
    assert events[-1] == ("holdout", selected.model_id)
    assert result.selected_window_id == selected.window_id
    assert result.holdout.selected_window_id == selected.window_id
    assert 999.0 not in result.aggregate_oos_returns


def test_aggregate_metrics_contain_only_window_test_observations() -> None:
    data = _data()
    plan = _plan()
    source, config = _inputs(data)
    trace: Dict[str, List] = {}
    optimizer, evaluator = _callbacks(trace)

    result = WalkForwardValidator(plan).run(
        data,
        source_data=source,
        base_config=config,
        optimizer=optimizer,
        evaluator=evaluator,
    )

    expected = tuple(
        value
        for window in WalkForwardValidator(plan).build_windows(len(data))
        for value in data.iloc[window.test_start : window.test_end]["return"]
    )
    assert result.aggregate_oos_returns == expected
    assert result.aggregate_oos_metrics["observation_count"] == len(expected)
    assert not set(result.aggregate_oos_returns).intersection(
        data.iloc[-plan.holdout_size :]["return"]
    )


def test_unannualized_return_to_volatility_does_not_scale_with_observation_count() -> None:
    short = EvaluationOutcome((0.01, -0.005)).metrics
    repeated = EvaluationOutcome((0.01, -0.005, 0.01, -0.005)).metrics

    assert short["return_to_volatility"] == pytest.approx(repeated["return_to_volatility"])
    assert "sharpe_ratio" not in short


def test_plan_rejects_mislabeled_observation_count_scaled_sharpe_metric() -> None:
    with pytest.raises(ValueError, match="unsupported selection metric"):
        _plan(selection_metric="sharpe_ratio")


def test_seeded_validation_is_byte_deterministic() -> None:
    data = _data()
    source, config = _inputs(data)

    def run_once():
        trace: Dict[str, List] = {}
        optimizer, evaluator = _callbacks(trace)
        return WalkForwardValidator(_plan(root_seed=123)).run(
            data,
            source_data=source,
            base_config=config,
            optimizer=optimizer,
            evaluator=evaluator,
        )

    first = run_once()
    second = run_once()
    assert first == second
    assert first.to_bytes() == second.to_bytes()
    assert first.evidence_digest() == second.evidence_digest()


def test_result_rejects_manual_oos_seed_and_holdout_evidence_tampering() -> None:
    data = _data()
    source, config = _inputs(data)
    optimizer, evaluator = _callbacks({})
    result = WalkForwardValidator(_plan()).run(
        data,
        source_data=source,
        base_config=config,
        optimizer=optimizer,
        evaluator=evaluator,
    )

    with pytest.raises(ValueError, match="only ordered window OOS"):
        replace(result, aggregate_oos_returns=(999.0,))
    changed_window = replace(result.windows[0], optimizer_seed=0)
    with pytest.raises(ValueError, match="window seed evidence"):
        replace(result, windows=(changed_window,) + result.windows[1:])
    with pytest.raises(ValueError, match="holdout partition identity"):
        replace(result, holdout=replace(result.holdout, partition_id="different"))


@pytest.mark.parametrize("failure", ["optimizer", "evaluator"])
def test_callback_failure_aborts_instead_of_skipping_window(failure: str) -> None:
    data = _data()
    source, config = _inputs(data)
    optimizer, evaluator = _callbacks({})

    def failed_optimizer(*_args):
        raise RuntimeError("fit failed")

    def failed_evaluator(*_args):
        raise RuntimeError("score failed")

    with pytest.raises(WalkForwardValidationError, match=f"{failure} failed"):
        WalkForwardValidator(_plan()).run(
            data,
            source_data=source,
            base_config=config,
            optimizer=failed_optimizer if failure == "optimizer" else optimizer,
            evaluator=failed_evaluator if failure == "evaluator" else evaluator,
        )


def test_empty_or_invalid_callback_outcome_fails_closed() -> None:
    data = _data()
    source, config = _inputs(data)
    _optimizer, evaluator = _callbacks({})

    with pytest.raises(WalkForwardValidationError, match="invalid or empty"):
        WalkForwardValidator(_plan()).run(
            data,
            source_data=source,
            base_config=config,
            optimizer=lambda *_args: None,
            evaluator=evaluator,
        )


def test_insufficient_events_fail_instead_of_shrinking_windows() -> None:
    plan = _plan()
    count = plan.minimum_development_size + plan.holdout_size - 1
    with pytest.raises(WalkForwardValidationError, match="insufficient events"):
        WalkForwardValidator(plan).build_windows(count)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda frame: frame.iloc[::-1], "sorted"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]).sort_index(), "unique"),
        (lambda frame: frame.tz_localize(None), "timezone-aware"),
        (lambda frame: frame.assign(feature=np.nan), "finite"),
    ],
)
def test_invalid_data_boundaries_fail_before_callbacks(mutate, match: str) -> None:
    data = _data()
    invalid = mutate(data)
    source = HashedInput("data", "invalid", digest_dataframe(data))
    config = hashed_json_input("config", "base", {})
    optimizer, evaluator = _callbacks({})
    with pytest.raises(ValueError, match=match):
        WalkForwardValidator(_plan()).run(
            invalid,
            source_data=source,
            base_config=config,
            optimizer=optimizer,
            evaluator=evaluator,
        )


def test_mismatched_source_hash_and_config_kind_are_rejected() -> None:
    data = _data()
    source, config = _inputs(data)
    optimizer, evaluator = _callbacks({})
    wrong_source = HashedInput("data", source.identifier, digest_dataframe(data.iloc[:-1]))
    with pytest.raises(ValueError, match="does not match"):
        WalkForwardValidator(_plan()).run(
            data,
            source_data=wrong_source,
            base_config=config,
            optimizer=optimizer,
            evaluator=evaluator,
        )
    with pytest.raises(ValueError, match="base_config"):
        WalkForwardValidator(_plan()).run(
            data,
            source_data=source,
            base_config=hashed_json_input("data", "not-config", {}),
            optimizer=optimizer,
            evaluator=evaluator,
        )
