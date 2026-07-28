"""Immutable and fail-closed backtest manifest contract tests."""

import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from robo_trader.backtesting.manifest import (
    BacktestRunManifest,
    DataAssumptions,
    DatasetPartition,
    EnvironmentMetadata,
    ExecutionAssumptions,
    PackageVersion,
    ResultArtifact,
    SeedAssignment,
    SeedPolicy,
)
from robo_trader.backtesting.provenance import ContentDigest, hashed_json_input


def _digest(label: str) -> ContentDigest:
    return hashed_json_input("data", label, {"label": label}).digest


def _manifest(*, approval_eligible: bool = True, errors=()) -> BacktestRunManifest:
    timezone = ZoneInfo("America/New_York")
    start = datetime(2020, 1, 1, tzinfo=timezone)
    split = datetime(2023, 1, 1, tzinfo=timezone)
    end = datetime(2024, 1, 1, tzinfo=timezone)
    inputs = (
        hashed_json_input("model", "model-set:none", {"models": []}),
        hashed_json_input("code", "git-tree:abc123", {"git_tree": "abc123"}),
        hashed_json_input("data", "prices-v1", {"dataset": "prices-v1"}),
        hashed_json_input("config", "strategy-config-v2", {"lookback": 20}),
    )
    partitions = (
        DatasetPartition(
            "train-2020-2022",
            "split-v1",
            "prices-v1",
            "train",
            start,
            split,
            _digest("train"),
            False,
        ),
        DatasetPartition(
            "holdout-2023",
            "split-v1",
            "prices-v1",
            "holdout",
            split,
            end,
            _digest("holdout"),
            True,
        ),
    )
    lock = hashed_json_input("package-lock", "requirements.lock", {"pandas": "3.0.0"})
    environment = EnvironmentMetadata(
        python_version="3.12.13",
        implementation="cpython",
        platform_id="macOS-arm64",
        packages=(PackageVersion("numpy", "2.4.1"), PackageVersion("pandas", "3.0.0")),
        package_lock=lock,
    )
    return BacktestRunManifest.create(
        run_id="candidate-42-holdout-1",
        created_at=datetime(2026, 7, 28, 12, 0, tzinfo=ZoneInfo("UTC")),
        strategy_id="mean-reversion",
        strategy_version="2.1.0",
        inputs=reversed(inputs),
        seed_policy=SeedPolicy.from_root(42, ["execution", "strategy"]),
        data_assumptions=DataAssumptions(
            timezone="America/New_York",
            calendar_id="XNYS-v2026",
            session_policy_id="regular-session-only-v1",
            bar_interval_id="1d-close-v1",
            start_at=start,
            end_at=end,
            corporate_action_policy_id="split-dividend-explicit-v1",
            price_adjustment_policy_id="unadjusted-ohlcv-v1",
            missing_quote_policy_id="fail-closed-v1",
        ),
        partitions=reversed(partitions),
        execution_assumptions=ExecutionAssumptions(
            commission_policy_id="per-share-minimum-v1",
            slippage_policy_id="owned-pcg64-adverse-v1",
            fill_policy_id="next-exact-bar-open-v1",
            market_impact_policy_id="participation-impact-v1",
            partial_fill_policy_id="carry-remainder-v1",
            finalization_policy_id="liquidate",
        ),
        environment=environment,
        approval_eligible=approval_eligible,
        recorded_errors=errors,
        result_id="result-candidate-42",
        result_format_id="canonical-json-v1",
        result_payload=b'{"final_equity":"101000.00"}',
    )


def test_manifest_is_deterministic_canonical_and_immutable() -> None:
    first = _manifest()
    second = _manifest()

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.manifest_digest() == second.manifest_digest()
    assert BacktestRunManifest.from_json(first.to_json()) == first
    assert [item.kind for item in first.inputs] == ["code", "config", "data", "model"]
    assert [item.partition_id for item in first.partitions] == [
        "holdout-2023",
        "train-2020-2022",
    ]
    with pytest.raises(FrozenInstanceError):
        first.run_id = "changed"


def test_result_content_and_input_linkage_are_verified() -> None:
    manifest = _manifest()
    payload = b'{"final_equity":"101000.00"}'

    manifest.verify_result(payload)
    manifest.validate_for_approval(payload)
    with pytest.raises(ValueError, match="result content"):
        manifest.verify_result(b'{"final_equity":"0"}')
    with pytest.raises(ValueError, match="linkage"):
        replace(
            manifest,
            result=replace(
                manifest.result,
                input_manifest_digest=ContentDigest("sha256", "0" * 64, 1),
            ),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        replace(
            manifest,
            result=ResultArtifact(
                result_id="empty",
                format_id="canonical-json-v1",
                content_digest=ContentDigest(
                    "sha256",
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    0,
                ),
                input_manifest_digest=manifest.result.input_manifest_digest,
            ),
        )


def test_any_provenance_change_invalidates_existing_result_link() -> None:
    manifest = _manifest()
    changed_input = hashed_json_input("config", "strategy-config-v3", {"lookback": 21})
    changed_inputs = tuple(item for item in manifest.inputs if item.kind != "config") + (
        changed_input,
    )

    with pytest.raises(ValueError, match="linkage"):
        replace(manifest, inputs=changed_inputs)
    with pytest.raises(ValueError, match="linkage"):
        replace(manifest, result=replace(manifest.result, result_id="different-result"))
    with pytest.raises(ValueError, match="linkage"):
        replace(manifest, result=replace(manifest.result, format_id="different-format"))


def test_manifest_requires_hashed_data_config_code_and_model_inputs() -> None:
    manifest = _manifest()
    without_model = tuple(item for item in manifest.inputs if item.kind != "model")

    with pytest.raises(ValueError, match="data/config/code/model"):
        replace(manifest, inputs=without_model)


def test_manifest_requires_one_untouched_nonoverlapping_holdout() -> None:
    manifest = _manifest()
    train = next(partition for partition in manifest.partitions if partition.role == "train")
    holdout = next(partition for partition in manifest.partitions if partition.role == "holdout")

    with pytest.raises(ValueError, match="one untouched holdout"):
        replace(manifest, partitions=(train,))
    with pytest.raises(ValueError, match="training partition"):
        replace(manifest, partitions=(holdout,))
    overlapping = replace(holdout, start_at=train.end_at.replace(year=2022))
    with pytest.raises(ValueError, match="must not overlap"):
        replace(manifest, partitions=(train, overlapping))


def test_partition_must_link_to_known_data_input_and_declared_window() -> None:
    manifest = _manifest()
    holdout = next(partition for partition in manifest.partitions if partition.role == "holdout")

    with pytest.raises(ValueError, match="unknown data input"):
        replace(
            manifest,
            partitions=tuple(
                (
                    replace(partition, source_data_input_id="unknown")
                    if partition == holdout
                    else partition
                )
                for partition in manifest.partitions
            ),
        )
    with pytest.raises(ValueError, match="outside"):
        replace(
            manifest,
            data_assumptions=replace(
                manifest.data_assumptions,
                end_at=datetime(2023, 6, 1, tzinfo=ZoneInfo("America/New_York")),
            ),
        )


def test_seed_policy_is_explicit_repeatable_and_rejects_mismatch() -> None:
    policy = SeedPolicy.from_root(99, ["strategy", "execution"])

    assert policy.seed_for("strategy") == SeedPolicy.derive_seed(99, "strategy")
    assert policy == SeedPolicy.from_root(99, ["execution", "strategy"])
    with pytest.raises(ValueError, match="assignment mismatch"):
        SeedPolicy(
            root_seed=99,
            generator_id="numpy-pcg64",
            derivation_id="sha256-component-v1",
            assignments=(SeedAssignment("strategy", 1),),
        )


def test_timezone_session_and_execution_policy_are_mandatory() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="declared IANA timezone"):
        replace(
            manifest.data_assumptions,
            start_at=datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
        )
    with pytest.raises(ValueError, match="non-empty"):
        replace(manifest.execution_assumptions, fill_policy_id="")
    with pytest.raises(ValueError, match="finalization"):
        replace(manifest.execution_assumptions, finalization_policy_id="invented")


def test_recorded_error_manifest_is_never_approval_eligible() -> None:
    manifest = _manifest(approval_eligible=False, errors=("strategy failed",))
    with pytest.raises(ValueError, match="not approval eligible"):
        manifest.validate_for_approval(b'{"final_equity":"101000.00"}')
    with pytest.raises(ValueError, match="recorded errors"):
        replace(manifest, approval_eligible=True)


def test_deserialization_rejects_incomplete_nonfinite_and_noncanonical_json() -> None:
    manifest = _manifest()
    value = manifest.to_dict()
    value.pop("result")
    with pytest.raises(ValueError, match="keys mismatch"):
        BacktestRunManifest.from_json(json.dumps(value, sort_keys=True, separators=(",", ":")))

    nonfinite = manifest.to_json().replace('"approval_eligible":true', '"approval_eligible":NaN')
    with pytest.raises(ValueError, match="non-finite"):
        BacktestRunManifest.from_json(nonfinite)

    with pytest.raises(ValueError, match="canonical form"):
        BacktestRunManifest.from_json(json.dumps(manifest.to_dict(), indent=2))


def test_deserialization_rejects_tampered_result_and_unknown_fields() -> None:
    manifest = _manifest()
    tampered = manifest.to_dict()
    tampered["strategy_version"] = "2.2.0"
    payload = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="linkage"):
        BacktestRunManifest.from_json(payload)

    unknown = manifest.to_dict()
    unknown["unreviewed"] = True
    with pytest.raises(ValueError, match="extra=.*unreviewed"):
        BacktestRunManifest.from_json(json.dumps(unknown, sort_keys=True, separators=(",", ":")))


def test_environment_packages_and_lock_are_explicit_and_sorted() -> None:
    manifest = _manifest()
    assert [package.name for package in manifest.environment.packages] == ["numpy", "pandas"]
    assert manifest.environment.package_lock.kind == "package-lock"
    with pytest.raises(ValueError, match="package-lock"):
        replace(
            manifest.environment,
            package_lock=hashed_json_input("config", "not-a-lock", {}),
        )
