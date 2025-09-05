"""Model registry for managing deployed models in RoboTrader.

This module provides model versioning, A/B testing, and deployment management.
"""

import asyncio
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class ModelRegistry:
    """Central registry for model deployment and versioning."""

    def __init__(self, registry_dir: Optional[Path] = None, max_versions: int = 10):
        """Initialize model registry.

        Args:
            registry_dir: Directory for registry storage
            max_versions: Maximum versions to keep per model
        """
        self.registry_dir = registry_dir or Path("model_registry")
        self.registry_dir.mkdir(exist_ok=True)

        self.max_versions = max_versions
        self.deployed_models: Dict[str, Dict] = {}
        self.model_versions: Dict[str, List[Dict]] = {}
        self.ab_tests: Dict[str, Dict] = {}

        # Load registry state
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry state from disk."""
        registry_file = self.registry_dir / "registry.json"

        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    state = json.load(f)

                self.deployed_models = state.get("deployed_models", {})
                self.model_versions = state.get("model_versions", {})
                self.ab_tests = state.get("ab_tests", {})

                logger.info(
                    "Registry loaded",
                    n_deployed=len(self.deployed_models),
                    n_versions=sum(len(v) for v in self.model_versions.values()),
                )
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save registry state to disk."""
        registry_file = self.registry_dir / "registry.json"

        state = {
            "deployed_models": self.deployed_models,
            "model_versions": self.model_versions,
            "ab_tests": self.ab_tests,
            "updated_at": datetime.now().isoformat(),
        }

        try:
            with open(registry_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

            logger.debug("Registry saved")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    async def register_model(
        self,
        model_info: Dict,
        model_name: str,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a new model version.

        Args:
            model_info: Model information dictionary
            model_name: Name for the model
            version: Version string (auto-generated if not provided)
            tags: Optional tags for the model

        Returns:
            Version identifier
        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        # Create model directory
        model_dir = self.registry_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Save model files
        version_dir = model_dir / version
        version_dir.mkdir(exist_ok=True)

        model_file = version_dir / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_info, f)

        # Create version metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "model_type": model_info.get("model_type"),
            "metrics": model_info.get("metrics", {}),
            "features": model_info.get("features", []),
            "tags": tags or [],
            "status": "registered",
        }

        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []

        self.model_versions[model_name].append(metadata)

        # Cleanup old versions
        await self._cleanup_old_versions(model_name)

        self._save_registry()

        logger.info(
            "Model registered", model_name=model_name, version=version, metrics=metadata["metrics"]
        )

        return version

    async def _cleanup_old_versions(self, model_name: str) -> None:
        """Remove old model versions beyond max_versions.

        Args:
            model_name: Name of the model
        """
        if model_name not in self.model_versions:
            return

        versions = self.model_versions[model_name]

        if len(versions) > self.max_versions:
            # Sort by registration time
            versions.sort(key=lambda x: x["registered_at"])

            # Remove oldest versions
            to_remove = versions[: -self.max_versions]

            for version_info in to_remove:
                version = version_info["version"]
                version_dir = self.registry_dir / model_name / version

                if version_dir.exists():
                    shutil.rmtree(version_dir)
                    logger.info("Removed old version", model_name=model_name, version=version)

            # Update registry
            self.model_versions[model_name] = versions[-self.max_versions :]

    async def deploy_model(
        self, model_name: str, version: Optional[str] = None, environment: str = "production"
    ) -> bool:
        """Deploy a model version to an environment.

        Args:
            model_name: Name of the model
            version: Version to deploy (latest if not specified)
            environment: Deployment environment

        Returns:
            Success status
        """
        if model_name not in self.model_versions:
            logger.error(f"Model {model_name} not found in registry")
            return False

        versions = self.model_versions[model_name]

        if version is None:
            # Deploy latest version
            version_info = versions[-1]
            version = version_info["version"]
        else:
            # Find specific version
            version_info = None
            for v in versions:
                if v["version"] == version:
                    version_info = v
                    break

            if version_info is None:
                logger.error(f"Version {version} not found for model {model_name}")
                return False

        # Load model
        model_file = self.registry_dir / model_name / version / "model.pkl"

        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return False

        with open(model_file, "rb") as f:
            # Security: Only load trusted model files from our own system
            # In production, consider using joblib or safer serialization
            # Security: Only loading trusted model files from our own system
            try:
                model_info = pickle.load(f)  # nosec B301 - Trusted file from our system
            except (pickle.PickleError, EOFError, ImportError) as e:
                logger.error(f"Failed to load model file {model_file}: {e}")
                return False

        # Update deployment status
        deployment = {
            "model_name": model_name,
            "version": version,
            "environment": environment,
            "deployed_at": datetime.now().isoformat(),
            "model_info": model_info,
            "metadata": version_info,
        }

        if environment not in self.deployed_models:
            self.deployed_models[environment] = {}

        self.deployed_models[environment][model_name] = deployment

        # Update version status
        version_info["status"] = f"deployed_{environment}"

        self._save_registry()

        logger.info(
            "Model deployed", model_name=model_name, version=version, environment=environment
        )

        return True

    async def get_deployed_model(
        self, model_name: str, environment: str = "production"
    ) -> Optional[Dict]:
        """Get currently deployed model.

        Args:
            model_name: Name of the model
            environment: Deployment environment

        Returns:
            Deployed model info or None
        """
        if environment not in self.deployed_models:
            return None

        if model_name not in self.deployed_models[environment]:
            return None

        deployment = self.deployed_models[environment][model_name]
        return deployment["model_info"]

    async def rollback_model(self, model_name: str, environment: str = "production") -> bool:
        """Rollback to previous model version.

        Args:
            model_name: Name of the model
            environment: Deployment environment

        Returns:
            Success status
        """
        if model_name not in self.model_versions:
            logger.error(f"Model {model_name} not found")
            return False

        versions = self.model_versions[model_name]

        # Find current deployment
        current_version = None
        if environment in self.deployed_models:
            if model_name in self.deployed_models[environment]:
                current_version = self.deployed_models[environment][model_name]["version"]

        if current_version is None:
            logger.error(f"No current deployment found for {model_name}")
            return False

        # Find previous version
        previous_version = None
        for i, v in enumerate(versions):
            if v["version"] == current_version and i > 0:
                previous_version = versions[i - 1]["version"]
                break

        if previous_version is None:
            logger.error(f"No previous version found for {model_name}")
            return False

        # Deploy previous version
        success = await self.deploy_model(model_name, previous_version, environment)

        if success:
            logger.info(
                "Model rolled back",
                model_name=model_name,
                from_version=current_version,
                to_version=previous_version,
                environment=environment,
            )

        return success

    async def start_ab_test(
        self,
        test_name: str,
        model_a: Dict[str, str],  # {"name": "model_name", "version": "v1"}
        model_b: Dict[str, str],
        traffic_split: float = 0.5,  # Fraction of traffic to model B
        duration_hours: int = 24,
    ) -> bool:
        """Start an A/B test between two models.

        Args:
            test_name: Name for the test
            model_a: Model A configuration
            model_b: Model B configuration
            traffic_split: Fraction of traffic for model B
            duration_hours: Test duration in hours

        Returns:
            Success status
        """
        # Validate models exist
        for model in [model_a, model_b]:
            name = model["name"]
            version = model.get("version")

            if name not in self.model_versions:
                logger.error(f"Model {name} not found")
                return False

            if version:
                versions = [v["version"] for v in self.model_versions[name]]
                if version not in versions:
                    logger.error(f"Version {version} not found for {name}")
                    return False

        # Create A/B test
        self.ab_tests[test_name] = {
            "test_name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "started_at": datetime.now().isoformat(),
            "duration_hours": duration_hours,
            "status": "active",
            "metrics_a": [],
            "metrics_b": [],
        }

        self._save_registry()

        logger.info(
            "A/B test started",
            test_name=test_name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
        )

        return True

    async def get_ab_test_model(self, test_name: str) -> Optional[Dict]:
        """Get model for A/B test based on traffic split.

        Args:
            test_name: Name of the A/B test

        Returns:
            Selected model info or None
        """
        if test_name not in self.ab_tests:
            return None

        test = self.ab_tests[test_name]

        if test["status"] != "active":
            return None

        # Check if test has expired
        started_at = datetime.fromisoformat(test["started_at"])
        elapsed_hours = (datetime.now() - started_at).total_seconds() / 3600

        if elapsed_hours > test["duration_hours"]:
            test["status"] = "completed"
            self._save_registry()
            return None

        # Select model based on traffic split
        import random

        use_model_b = random.random() < test["traffic_split"]

        if use_model_b:
            model_config = test["model_b"]
        else:
            model_config = test["model_a"]

        # Load model
        model_name = model_config["name"]
        version = model_config.get("version")

        if version is None:
            # Use latest version
            version = self.model_versions[model_name][-1]["version"]

        model_file = self.registry_dir / model_name / version / "model.pkl"

        with open(model_file, "rb") as f:
            # Security: Only load trusted model files from our own system
            try:
                model_info = pickle.load(f)  # nosec B301 - Trusted file from our system
            except (pickle.PickleError, EOFError, ImportError) as e:
                logger.error(f"Failed to load model file {model_file}: {e}")
                return None

        model_info["ab_test"] = test_name
        model_info["ab_variant"] = "B" if use_model_b else "A"

        return model_info

    async def record_ab_test_metric(
        self, test_name: str, variant: str, metric_name: str, metric_value: float
    ) -> None:
        """Record metric for A/B test.

        Args:
            test_name: Name of the test
            variant: "A" or "B"
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        if test_name not in self.ab_tests:
            return

        test = self.ab_tests[test_name]

        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "metric_value": metric_value,
        }

        if variant == "A":
            test["metrics_a"].append(metric)
        else:
            test["metrics_b"].append(metric)

        self._save_registry()

    def get_ab_test_results(self, test_name: str) -> Optional[Dict]:
        """Get results of an A/B test.

        Args:
            test_name: Name of the test

        Returns:
            Test results or None
        """
        if test_name not in self.ab_tests:
            return None

        test = self.ab_tests[test_name]

        # Calculate metrics
        def calculate_stats(metrics: List[Dict]) -> Dict:
            if not metrics:
                return {"count": 0, "mean": 0, "std": 0}

            df = pd.DataFrame(metrics)

            if df.empty:
                return {"count": 0, "mean": 0, "std": 0}

            grouped = df.groupby("metric_name")["metric_value"]

            stats = {}
            for name, group in grouped:
                stats[name] = {
                    "count": len(group),
                    "mean": group.mean(),
                    "std": group.std(),
                    "min": group.min(),
                    "max": group.max(),
                }

            return stats

        results = {
            "test_name": test_name,
            "status": test["status"],
            "model_a": test["model_a"],
            "model_b": test["model_b"],
            "traffic_split": test["traffic_split"],
            "started_at": test["started_at"],
            "metrics_a": calculate_stats(test["metrics_a"]),
            "metrics_b": calculate_stats(test["metrics_b"]),
        }

        # Determine winner if test is completed
        if test["status"] == "completed" and results["metrics_a"] and results["metrics_b"]:
            # Compare primary metric (assume first metric is primary)
            primary_metrics_a = list(results["metrics_a"].values())
            primary_metrics_b = list(results["metrics_b"].values())

            if primary_metrics_a and primary_metrics_b:
                mean_a = primary_metrics_a[0].get("mean", 0)
                mean_b = primary_metrics_b[0].get("mean", 0)

                if mean_a > mean_b:
                    results["winner"] = "A"
                    results["improvement"] = (mean_a - mean_b) / mean_b if mean_b != 0 else 0
                else:
                    results["winner"] = "B"
                    results["improvement"] = (mean_b - mean_a) / mean_a if mean_a != 0 else 0

        return results

    def get_registry_summary(self) -> Dict:
        """Get summary of registry state.

        Returns:
            Registry summary
        """
        summary = {
            "total_models": len(self.model_versions),
            "total_versions": sum(len(v) for v in self.model_versions.values()),
            "deployed_models": {},
            "active_ab_tests": [],
            "model_details": {},
        }

        # Deployed models by environment
        for env, models in self.deployed_models.items():
            summary["deployed_models"][env] = list(models.keys())

        # Active A/B tests
        for test_name, test in self.ab_tests.items():
            if test["status"] == "active":
                summary["active_ab_tests"].append(test_name)

        # Model details
        for model_name, versions in self.model_versions.items():
            latest = versions[-1] if versions else None
            summary["model_details"][model_name] = {
                "n_versions": len(versions),
                "latest_version": latest["version"] if latest else None,
                "latest_metrics": latest.get("metrics", {}) if latest else {},
                "tags": latest.get("tags", []) if latest else [],
            }

        return summary
