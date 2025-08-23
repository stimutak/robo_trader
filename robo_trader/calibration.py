"""
Calibration metrics and Brier score tracking for model performance.
Ensures LLM predictions are well-calibrated with actual outcomes.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

from .logger import get_logger
from .database import TradingDatabase

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    period_days: int
    total_decisions: int
    brier_score: float
    reliability: float  # Slope of reliability diagram
    resolution: float  # Ability to discriminate
    calibration_error: float  # Average calibration error
    sharpness: float  # Confidence in predictions
    
    # Bucketed analysis
    conviction_buckets: Dict[str, Dict]  # Stats by conviction level
    outcome_distribution: Dict[str, int]  # Wins/losses/timeouts
    
    # Performance metrics
    actual_win_rate: float
    predicted_win_rate: float
    ev_accuracy: float  # How accurate were EV predictions
    
    def is_well_calibrated(self) -> bool:
        """Check if model is well calibrated."""
        return (
            self.brier_score <= 0.20 and  # Good Brier score
            0.8 <= self.reliability <= 1.2 and  # Good reliability
            abs(self.calibration_error) <= 0.10  # Low calibration error
        )


class CalibrationTracker:
    """
    Track and calculate calibration metrics for LLM decisions.
    Implements Brier score and reliability analysis.
    """
    
    def __init__(self, database: Optional[TradingDatabase] = None):
        """
        Initialize calibration tracker.
        
        Args:
            database: Optional database for persistence
        """
        self.db = database or TradingDatabase()
        self.recent_predictions: List[Tuple[float, bool]] = []  # (predicted_prob, actual_outcome)
        
        logger.info("CalibrationTracker initialized")
    
    def calculate_brier_score(
        self,
        predictions: List[float],
        outcomes: List[bool]
    ) -> float:
        """
        Calculate Brier score for probability calibration.
        
        Brier Score = mean((prediction - outcome)^2)
        Lower is better, 0 is perfect, 1 is worst.
        
        Args:
            predictions: Predicted probabilities (0-1)
            outcomes: Actual outcomes (True/False)
            
        Returns:
            Brier score
        """
        if len(predictions) != len(outcomes) or len(predictions) == 0:
            return 1.0
        
        predictions = np.array(predictions)
        outcomes = np.array(outcomes, dtype=float)
        
        brier = np.mean((predictions - outcomes) ** 2)
        
        return float(brier)
    
    def calculate_reliability(
        self,
        predictions: List[float],
        outcomes: List[bool],
        n_bins: int = 10
    ) -> Tuple[float, Dict[str, List]]:
        """
        Calculate reliability (calibration plot slope).
        
        Perfect reliability = 1.0 (45-degree line)
        
        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes
            n_bins: Number of bins for grouping
            
        Returns:
            Tuple of (reliability_slope, bin_data)
        """
        if len(predictions) < 10:
            return 1.0, {}
        
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_data = {
            'centers': [],
            'predicted': [],
            'actual': [],
            'counts': []
        }
        
        # Calculate actual vs predicted for each bin
        for i in range(n_bins):
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
            
            if np.sum(mask) > 0:
                bin_predictions = np.array(predictions)[mask]
                bin_outcomes = np.array(outcomes)[mask]
                
                bin_data['centers'].append(bin_centers[i])
                bin_data['predicted'].append(np.mean(bin_predictions))
                bin_data['actual'].append(np.mean(bin_outcomes))
                bin_data['counts'].append(len(bin_predictions))
        
        # Calculate slope (reliability)
        if len(bin_data['predicted']) >= 2:
            # Linear regression for slope
            x = np.array(bin_data['predicted'])
            y = np.array(bin_data['actual'])
            
            # Weighted by counts
            weights = np.array(bin_data['counts'])
            weights = weights / np.sum(weights)
            
            # Calculate weighted slope
            mean_x = np.average(x, weights=weights)
            mean_y = np.average(y, weights=weights)
            
            numerator = np.sum(weights * (x - mean_x) * (y - mean_y))
            denominator = np.sum(weights * (x - mean_x) ** 2)
            
            if denominator > 0:
                slope = numerator / denominator
            else:
                slope = 1.0
        else:
            slope = 1.0
        
        return float(slope), bin_data
    
    def calculate_resolution(
        self,
        predictions: List[float],
        outcomes: List[bool]
    ) -> float:
        """
        Calculate resolution (ability to discriminate).
        
        Higher is better - measures spread of predictions.
        
        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes
            
        Returns:
            Resolution score
        """
        if len(predictions) == 0:
            return 0.0
        
        base_rate = np.mean(outcomes)
        
        # Group by prediction levels
        unique_preds = np.unique(np.round(predictions, 2))
        
        resolution = 0.0
        for pred in unique_preds:
            mask = np.abs(np.array(predictions) - pred) < 0.01
            if np.sum(mask) > 0:
                weight = np.sum(mask) / len(predictions)
                outcome_rate = np.mean(np.array(outcomes)[mask])
                resolution += weight * (outcome_rate - base_rate) ** 2
        
        return float(resolution)
    
    def analyze_decisions(
        self,
        days: int = 30,
        min_decisions: int = 20
    ) -> Optional[CalibrationResult]:
        """
        Analyze LLM decisions for calibration.
        
        Args:
            days: Number of days to analyze
            min_decisions: Minimum decisions required
            
        Returns:
            CalibrationResult or None if insufficient data
        """
        # Get decision data from database
        decisions = self.db.get_calibration_data(days=days)
        
        if len(decisions) < min_decisions:
            logger.warning(
                f"Insufficient decisions for calibration: "
                f"{len(decisions)} < {min_decisions}"
            )
            return None
        
        # Extract predictions and outcomes
        predictions = []
        outcomes = []
        ev_predictions = []
        ev_actuals = []
        
        conviction_buckets = {
            "0-30": {"predictions": [], "outcomes": []},
            "31-50": {"predictions": [], "outcomes": []},
            "51-70": {"predictions": [], "outcomes": []},
            "71-100": {"predictions": [], "outcomes": []}
        }
        
        outcome_dist = {"win": 0, "loss": 0, "scratch": 0, "timeout": 0}
        
        for decision in decisions:
            # Convert conviction to probability
            pred_prob = decision['conviction'] / 100.0
            predictions.append(pred_prob)
            
            # Determine actual outcome
            actual_outcome = decision['actual_outcome']
            is_win = actual_outcome == 'win'
            outcomes.append(is_win)
            
            # Track outcome distribution
            if actual_outcome in outcome_dist:
                outcome_dist[actual_outcome] += 1
            
            # Bucket by conviction
            conviction = decision['conviction']
            if conviction <= 30:
                bucket = "0-30"
            elif conviction <= 50:
                bucket = "31-50"
            elif conviction <= 70:
                bucket = "51-70"
            else:
                bucket = "71-100"
            
            conviction_buckets[bucket]["predictions"].append(pred_prob)
            conviction_buckets[bucket]["outcomes"].append(is_win)
            
            # Track EV accuracy
            if decision['expected_value_pct'] is not None and decision['actual_pnl'] is not None:
                ev_predictions.append(decision['expected_value_pct'])
                # Convert actual P&L to percentage (assuming 1% risk per trade)
                ev_actuals.append(decision['actual_pnl'] / 100)
        
        # Calculate metrics
        brier_score = self.calculate_brier_score(predictions, outcomes)
        reliability, bin_data = self.calculate_reliability(predictions, outcomes)
        resolution = self.calculate_resolution(predictions, outcomes)
        
        # Calculate calibration error
        calibration_error = np.mean(predictions) - np.mean(outcomes)
        
        # Calculate sharpness (standard deviation of predictions)
        sharpness = np.std(predictions) if predictions else 0
        
        # Calculate bucket statistics
        for bucket, data in conviction_buckets.items():
            if data["predictions"]:
                bucket_predictions = data["predictions"]
                bucket_outcomes = data["outcomes"]
                conviction_buckets[bucket] = {
                    "count": len(bucket_predictions),
                    "avg_prediction": np.mean(bucket_predictions),
                    "actual_win_rate": np.mean(bucket_outcomes),
                    "calibration_error": np.mean(bucket_predictions) - np.mean(bucket_outcomes)
                }
            else:
                conviction_buckets[bucket] = {
                    "count": 0,
                    "avg_prediction": 0,
                    "actual_win_rate": 0,
                    "calibration_error": 0
                }
        
        # Calculate EV accuracy
        ev_accuracy = 0
        if ev_predictions and ev_actuals:
            ev_errors = [abs(pred - actual) for pred, actual in zip(ev_predictions, ev_actuals)]
            ev_accuracy = 1.0 - min(1.0, np.mean(ev_errors) / 10)  # Normalize to 0-1
        
        result = CalibrationResult(
            period_days=days,
            total_decisions=len(decisions),
            brier_score=brier_score,
            reliability=reliability,
            resolution=resolution,
            calibration_error=calibration_error,
            sharpness=sharpness,
            conviction_buckets=conviction_buckets,
            outcome_distribution=outcome_dist,
            actual_win_rate=np.mean(outcomes),
            predicted_win_rate=np.mean(predictions),
            ev_accuracy=ev_accuracy
        )
        
        # Log summary
        logger.info(
            f"Calibration Analysis ({days} days, {len(decisions)} decisions): "
            f"Brier={brier_score:.3f}, Reliability={reliability:.2f}, "
            f"WinRate={result.actual_win_rate:.1%} vs Predicted={result.predicted_win_rate:.1%}"
        )
        
        # Save to database
        self._save_calibration_metrics(result)
        
        return result
    
    def _save_calibration_metrics(self, result: CalibrationResult):
        """Save calibration metrics to database."""
        metrics = {
            'period_days': result.period_days,
            'total_decisions': result.total_decisions,
            'trade_rate': result.total_decisions / max(result.period_days, 1),
            'win_rate': result.actual_win_rate,
            'avg_conviction': result.predicted_win_rate * 100,
            'brier_score': result.brier_score,
            'reliability': result.reliability,
            'resolution': result.resolution,
            'avg_ev_error': 1.0 - result.ev_accuracy,
            'sharpe_ratio': 0,  # Would calculate from returns
            'max_drawdown_pct': 0  # Would calculate from equity curve
        }
        
        self.db.save_calibration_metrics(metrics)
    
    def get_recommendation(self, result: CalibrationResult) -> str:
        """
        Get recommendation based on calibration analysis.
        
        Args:
            result: Calibration result
            
        Returns:
            Recommendation string
        """
        recommendations = []
        
        # Check Brier score
        if result.brier_score > 0.25:
            recommendations.append("Poor calibration: Consider retraining or adjusting conviction mapping")
        elif result.brier_score > 0.20:
            recommendations.append("Marginal calibration: Monitor closely and consider adjustments")
        
        # Check reliability
        if result.reliability < 0.8:
            recommendations.append("Underconfident: Model predictions are too conservative")
        elif result.reliability > 1.2:
            recommendations.append("Overconfident: Model predictions are too aggressive")
        
        # Check calibration error
        if abs(result.calibration_error) > 0.10:
            if result.calibration_error > 0:
                recommendations.append("Systematic overestimation: Reduce conviction scores")
            else:
                recommendations.append("Systematic underestimation: Increase conviction scores")
        
        # Check resolution
        if result.resolution < 0.01:
            recommendations.append("Low discrimination: Model struggles to differentiate opportunities")
        
        # Check conviction distribution
        high_conviction = result.conviction_buckets.get("71-100", {})
        if high_conviction.get("count", 0) < result.total_decisions * 0.1:
            recommendations.append("Too few high-conviction trades: Consider adjusting thresholds")
        
        if not recommendations:
            return "Model is well-calibrated. Continue monitoring performance."
        
        return " | ".join(recommendations)
    
    def track_prediction(self, predicted_prob: float, actual_outcome: bool):
        """
        Track a single prediction for running calibration.
        
        Args:
            predicted_prob: Predicted probability (0-1)
            actual_outcome: Actual outcome
        """
        self.recent_predictions.append((predicted_prob, actual_outcome))
        
        # Keep only last 1000 predictions
        if len(self.recent_predictions) > 1000:
            self.recent_predictions.pop(0)
        
        # Calculate running Brier score
        if len(self.recent_predictions) >= 20:
            predictions = [p[0] for p in self.recent_predictions[-100:]]
            outcomes = [p[1] for p in self.recent_predictions[-100:]]
            
            brier = self.calculate_brier_score(predictions, outcomes)
            
            if brier > 0.25:
                logger.warning(f"Running Brier score degraded: {brier:.3f}")