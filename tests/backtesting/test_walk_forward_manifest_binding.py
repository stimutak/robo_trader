"""Manifest binding tests for versioned walk-forward model evidence."""

from dataclasses import replace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from robo_trader.backtesting.manifest import (
    BacktestRunManifest,
    DataAssumptions,
    DatasetPartition,
    EnvironmentMetadata,
    ExecutionAssumptions,
    PackageVersion,
    SeedPolicy,
)
from robo_trader.backtesting.provenance import (
    HashedInput,
    digest_dataframe,
    hashed_json_input,
)
from robo_trader.backtesting.walk_forward import WalkForwardValidator
from tests.backtesting.test_walk_forward import _callbacks, _data, _inputs, _plan


def _result():
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
    return data, result


def _manifest(data: pd.DataFrame, result, **overrides) -> BacktestRunManifest:
    plan = result.plan
    holdout_start = len(data) - plan.holdout_size
    timezone = ZoneInfo("America/New_York")
    end = data.index[-1].to_pydatetime() + timedelta(days=1)
    model_id = overrides.pop("model_id", "walk-forward-model-bundle-v1")
    model_input = overrides.pop("model_input", result.as_model_input(model_id))
    inputs = (
        result.source_data,
        result.base_config,
        hashed_json_input("code", "git-tree-v1", {"tree": "abc123"}),
        model_input,
    )
    partitions = (
        DatasetPartition(
            partition_id="development",
            split_id=plan.split_id,
            source_data_input_id=result.source_data.identifier,
            role="train",
            start_at=data.index[0].to_pydatetime(),
            end_at=data.index[holdout_start].to_pydatetime(),
            content_digest=digest_dataframe(data.iloc[:holdout_start]),
            untouched=False,
        ),
        DatasetPartition(
            partition_id=plan.holdout_partition_id,
            split_id=plan.split_id,
            source_data_input_id=result.source_data.identifier,
            role="holdout",
            start_at=data.index[holdout_start].to_pydatetime(),
            end_at=end,
            content_digest=result.holdout.data_digest,
            untouched=True,
        ),
    )
    package_lock = hashed_json_input("package-lock", "requirements.lock", {"pandas": "3"})
    values = {
        "run_id": "walk-forward-candidate-1",
        "created_at": datetime(2026, 7, 28, tzinfo=ZoneInfo("UTC")),
        "strategy_id": "strategy-v1",
        "strategy_version": "1.0.0",
        "inputs": inputs,
        "seed_policy": result.seed_policy,
        "data_assumptions": DataAssumptions(
            timezone="America/New_York",
            calendar_id="XNYS-v2026",
            session_policy_id="regular-v1",
            bar_interval_id="1d-v1",
            start_at=data.index[0].to_pydatetime(),
            end_at=end,
            corporate_action_policy_id="explicit-v1",
            price_adjustment_policy_id="unadjusted-v1",
            missing_quote_policy_id="fail-closed-v1",
        ),
        "partitions": partitions,
        "execution_assumptions": ExecutionAssumptions(
            commission_policy_id="commission-v1",
            slippage_policy_id="slippage-v1",
            fill_policy_id="next-bar-v1",
            market_impact_policy_id="impact-v1",
            partial_fill_policy_id="carry-v1",
            finalization_policy_id="liquidate",
        ),
        "environment": EnvironmentMetadata(
            python_version="3.12.13",
            implementation="cpython",
            platform_id="test-platform",
            packages=(PackageVersion("pandas", "3.0.0"),),
            package_lock=package_lock,
        ),
        "approval_eligible": True,
        "recorded_errors": (),
        "result_id": "walk-forward-result-1",
        "result_format_id": "robotrader-walk-forward-v1-json",
        "result_payload": result.to_bytes(),
    }
    values.update(overrides)
    return BacktestRunManifest.create(**values)


def test_model_config_data_seed_holdout_and_result_bind_to_manifest() -> None:
    data, result = _result()
    manifest = _manifest(data, result)

    bound = result.bind_to_manifest(
        manifest, model_evidence_input_id="walk-forward-model-bundle-v1"
    )

    assert bound.manifest_digest == manifest.manifest_digest()
    assert bound.validation_result_digest == result.evidence_digest()
    assert bound.source_data_input_id == result.source_data.identifier
    assert bound.base_config_input_id == result.base_config.identifier
    assert bound.result_id == manifest.result.result_id


def test_manifest_binding_rejects_model_data_config_and_seed_mismatch() -> None:
    data, result = _result()
    manifest = _manifest(data, result)
    wrong_model = HashedInput(
        "model",
        "walk-forward-model-bundle-v1",
        hashed_json_input("model", "wrong", {"wrong": True}).digest,
    )
    with pytest.raises(ValueError, match="model evidence mismatch"):
        result.bind_to_manifest(
            _manifest(data, result, model_input=wrong_model),
            model_evidence_input_id="walk-forward-model-bundle-v1",
        )

    wrong_data = HashedInput(
        "data", result.source_data.identifier, digest_dataframe(data.iloc[:-1])
    )
    inputs = tuple(wrong_data if item.kind == "data" else item for item in manifest.inputs)
    with pytest.raises(ValueError, match="data evidence mismatch"):
        result.bind_to_manifest(
            _manifest(data, result, inputs=inputs),
            model_evidence_input_id="walk-forward-model-bundle-v1",
        )

    wrong_config = HashedInput(
        "config",
        result.base_config.identifier,
        hashed_json_input("config", "wrong", {"wrong": True}).digest,
    )
    config_inputs = tuple(
        wrong_config if item.kind == "config" else item for item in manifest.inputs
    )
    with pytest.raises(ValueError, match="config evidence mismatch"):
        result.bind_to_manifest(
            _manifest(data, result, inputs=config_inputs),
            model_evidence_input_id="walk-forward-model-bundle-v1",
        )

    wrong_seed = SeedPolicy.from_root(
        result.seed_policy.root_seed + 1,
        [assignment.component_id for assignment in result.seed_policy.assignments],
        result.seed_policy.generator_id,
    )
    with pytest.raises(ValueError, match="seed policy"):
        result.bind_to_manifest(
            _manifest(data, result, seed_policy=wrong_seed),
            model_evidence_input_id="walk-forward-model-bundle-v1",
        )


def test_manifest_binding_rejects_result_or_holdout_mismatch() -> None:
    data, result = _result()
    manifest = _manifest(data, result)
    with pytest.raises(ValueError, match="result content"):
        result.bind_to_manifest(
            _manifest(data, result, result_payload=b"different result"),
            model_evidence_input_id="walk-forward-model-bundle-v1",
        )

    holdout = next(partition for partition in manifest.partitions if partition.role == "holdout")
    wrong_holdout = replace(
        holdout,
        content_digest=digest_dataframe(data.iloc[-result.plan.holdout_size - 1 : -1]),
    )
    partitions = tuple(
        wrong_holdout if partition.role == "holdout" else partition
        for partition in manifest.partitions
    )
    with pytest.raises(ValueError, match="holdout content digest"):
        result.bind_to_manifest(
            _manifest(data, result, partitions=partitions),
            model_evidence_input_id="walk-forward-model-bundle-v1",
        )
