from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats


class CorrelationMethod(Enum):
    """Available correlation estimation methods."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    EWMA = "ewma"


@dataclass
class CorrelationChangePoint:
    """Represents a detected change in correlation."""

    date: pd.Timestamp
    correlation_before: float
    correlation_after: float
    z_statistic: float
    p_value: float
    direction: Literal["increase", "decrease"]


@dataclass
class RollingCorrelationResult:
    """Complete rolling correlation analysis results."""

    # Identifiers
    asset_x: str
    asset_y: str
    method: CorrelationMethod
    window: int

    # Time series
    correlation_series: pd.Series

    # Summary statistics
    mean_correlation: float
    std_correlation: float
    min_correlation: float
    max_correlation: float
    current_correlation: float
    current_percentile: float  # Where current sits in historical distribution

    # Regime classification
    current_regime: Literal["low", "normal", "high"]
    regime_series: pd.Series  # Classification over time

    # Change points
    change_points: list[CorrelationChangePoint]

    # Metadata
    n_observations: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


class RollingCorrelationTracker:
    """
    Computes and analyzes time-varying correlations between asset pairs.

    Supports Pearson, Spearman, and exponentially weighted correlation
    with regime detection and change point analysis.
    """

    # Default regime thresholds (percentiles)
    LOW_REGIME_THRESHOLD = 25
    HIGH_REGIME_THRESHOLD = 75

    def __init__(
        self,
        window: int = 60,
        method: CorrelationMethod = CorrelationMethod.PEARSON,
        min_periods: Optional[int] = None,
        ewma_lambda: float = 0.94,
        change_point_threshold: float = 0.05,
    ):
        """
        Initialize the correlation tracker.

        Args:
            window: Rolling window size in periods.
            method: Correlation method to use.
            min_periods: Minimum observations required. Defaults to window//2.
            ewma_lambda: Decay factor for EWMA (only used if method is EWMA).
            change_point_threshold: P-value threshold for change point detection.
        """
        if window < 10:
            raise ValueError(
                "Window must be at least 10 for reliable correlation estimates"
            )

        self.window = window
        self.method = method
        self.min_periods = min_periods or window // 2
        self.ewma_lambda = ewma_lambda
        self.change_point_threshold = change_point_threshold

    def _compute_rolling_pearson(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """
        Compute rolling Pearson correlation.

        Uses pandas built-in for efficiency and numerical stability.
        """
        return x.rolling(window=self.window, min_periods=self.min_periods).corr(y)

    def _compute_rolling_spearman(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """
        Compute rolling Spearman rank correlation.

        Applies Pearson correlation to ranks within each window.
        """

        def spearman_corr(window_x, window_y):
            # Handle the case where both are passed as a single 2D array
            return stats.spearmanr(window_x, window_y)[0]

        # Create a DataFrame for rolling application
        df = pd.DataFrame({"x": x, "y": y})

        result = df.rolling(window=self.window, min_periods=self.min_periods).apply(
            lambda w: stats.spearmanr(
                w.iloc[:, 0] if hasattr(w, "iloc") else w[: len(w) // 2],
                w.iloc[:, 1] if hasattr(w, "iloc") else w[len(w) // 2 :],
            )[0]
            if len(w) >= self.min_periods
            else np.nan,
            raw=False,
        )

        # Alternative: more efficient implementation using ranks
        ranks_x = x.rolling(window=self.window, min_periods=self.min_periods).apply(
            lambda w: pd.Series(w).rank().iloc[-1], raw=False
        )
        # This is complex; we use scipy directly for clarity

        correlations = []
        for i in range(len(x)):
            if i < self.min_periods - 1:
                correlations.append(np.nan)
            else:
                start = max(0, i - self.window + 1)
                window_x = x.iloc[start : i + 1]
                window_y = y.iloc[start : i + 1]
                if len(window_x) >= self.min_periods:
                    corr, _ = stats.spearmanr(window_x, window_y)
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)

        return pd.Series(correlations, index=x.index)

    def _compute_ewma_correlation(self, x: pd.Series, y: pd.Series) -> pd.Series:
        """
        Compute exponentially weighted moving average correlation.

        Uses the RiskMetrics methodology with configurable decay factor.
        """
        # Demean the series using expanding mean for each point
        # For EWMA, we typically assume zero mean or use centered returns
        x_centered = x - x.expanding().mean()
        y_centered = y - y.expanding().mean()

        # EWMA covariance and variances
        cov_xy = pd.Series(index=x.index, dtype=float)
        var_x = pd.Series(index=x.index, dtype=float)
        var_y = pd.Series(index=x.index, dtype=float)

        lambda_ = self.ewma_lambda

        # Initialize with first observation
        cov_xy.iloc[0] = x_centered.iloc[0] * y_centered.iloc[0]
        var_x.iloc[0] = x_centered.iloc[0] ** 2
        var_y.iloc[0] = y_centered.iloc[0] ** 2

        # Recursive update
        for i in range(1, len(x)):
            cov_xy.iloc[i] = (
                lambda_ * cov_xy.iloc[i - 1]
                + (1 - lambda_) * x_centered.iloc[i] * y_centered.iloc[i]
            )
            var_x.iloc[i] = (
                lambda_ * var_x.iloc[i - 1] + (1 - lambda_) * x_centered.iloc[i] ** 2
            )
            var_y.iloc[i] = (
                lambda_ * var_y.iloc[i - 1] + (1 - lambda_) * y_centered.iloc[i] ** 2
            )

        # Compute correlation
        correlation = cov_xy / np.sqrt(var_x * var_y)

        # First several values are unreliable
        correlation.iloc[: self.min_periods] = np.nan

        return correlation

    def _classify_regimes(self, correlation_series: pd.Series) -> pd.Series:
        """
        Classify each point into low/normal/high correlation regime.

        Uses historical percentiles as thresholds.
        """
        # Compute expanding percentiles
        regimes = pd.Series(index=correlation_series.index, dtype=object)

        for i in range(len(correlation_series)):
            if pd.isna(correlation_series.iloc[i]):
                regimes.iloc[i] = np.nan
                continue

            # Get history up to this point
            history = correlation_series.iloc[: i + 1].dropna()

            if len(history) < 20:  # Need minimum history for percentiles
                regimes.iloc[i] = "normal"
                continue

            current = correlation_series.iloc[i]
            percentile = stats.percentileofscore(history, current)

            if percentile <= self.LOW_REGIME_THRESHOLD:
                regimes.iloc[i] = "low"
            elif percentile >= self.HIGH_REGIME_THRESHOLD:
                regimes.iloc[i] = "high"
            else:
                regimes.iloc[i] = "normal"

        return regimes

    def _detect_change_points(
        self, correlation_series: pd.Series
    ) -> list[CorrelationChangePoint]:
        """
        Detect statistically significant changes in correlation level.

        Uses Fisher's Z-test to compare adjacent windows.
        """
        change_points = []

        # Minimum window for comparison
        comparison_window = self.window

        clean_series = correlation_series.dropna()

        if len(clean_series) < 2 * comparison_window:
            return change_points

        # Slide through series comparing adjacent windows
        for i in range(
            comparison_window, len(clean_series) - comparison_window, comparison_window
        ):
            window_before = clean_series.iloc[i - comparison_window : i]
            window_after = clean_series.iloc[i : i + comparison_window]

            r_before = window_before.mean()
            r_after = window_after.mean()

            # Fisher Z-transformation
            z_before = np.arctanh(np.clip(r_before, -0.999, 0.999))
            z_after = np.arctanh(np.clip(r_after, -0.999, 0.999))

            # Standard error of difference
            n1 = len(window_before)
            n2 = len(window_after)
            se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

            # Z-statistic
            z_stat = (z_after - z_before) / se

            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            if p_value < self.change_point_threshold:
                direction = "increase" if r_after > r_before else "decrease"

                change_points.append(
                    CorrelationChangePoint(
                        date=clean_series.index[i],
                        correlation_before=r_before,
                        correlation_after=r_after,
                        z_statistic=z_stat,
                        p_value=p_value,
                        direction=direction,
                    )
                )

        return change_points

    def analyze(
        self, data: pd.DataFrame, asset_x: str, asset_y: str
    ) -> RollingCorrelationResult:
        """
        Perform complete rolling correlation analysis for an asset pair.

        Args:
            data: DataFrame with asset returns (columns are assets).
            asset_x: Column name for first asset.
            asset_y: Column name for second asset.

        Returns:
            RollingCorrelationResult with full analysis.
        """
        if asset_x not in data.columns or asset_y not in data.columns:
            raise ValueError(f"Assets {asset_x} and/or {asset_y} not in data")

        x = data[asset_x].dropna()
        y = data[asset_y].dropna()

        # Align series
        aligned = pd.DataFrame({"x": x, "y": y}).dropna()
        x = aligned["x"]
        y = aligned["y"]

        if len(aligned) < self.window:
            raise ValueError(
                f"Insufficient data: {len(aligned)} observations, "
                f"need at least {self.window}"
            )

        # Compute correlation based on method
        if self.method == CorrelationMethod.PEARSON:
            corr_series = self._compute_rolling_pearson(x, y)
        elif self.method == CorrelationMethod.SPEARMAN:
            corr_series = self._compute_rolling_spearman(x, y)
        elif self.method == CorrelationMethod.EWMA:
            corr_series = self._compute_ewma_correlation(x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Clean correlation series
        corr_clean = corr_series.dropna()

        if len(corr_clean) == 0:
            raise ValueError("No valid correlation values computed")

        # Summary statistics
        mean_corr = corr_clean.mean()
        std_corr = corr_clean.std()
        min_corr = corr_clean.min()
        max_corr = corr_clean.max()
        current_corr = corr_clean.iloc[-1]
        current_pct = stats.percentileofscore(corr_clean, current_corr)

        # Regime classification
        regime_series = self._classify_regimes(corr_series)
        current_regime = regime_series.dropna().iloc[-1]

        # Change point detection
        change_points = self._detect_change_points(corr_series)

        return RollingCorrelationResult(
            asset_x=asset_x,
            asset_y=asset_y,
            method=self.method,
            window=self.window,
            correlation_series=corr_series,
            mean_correlation=mean_corr,
            std_correlation=std_corr,
            min_correlation=min_corr,
            max_correlation=max_corr,
            current_correlation=current_corr,
            current_percentile=current_pct,
            current_regime=current_regime,
            regime_series=regime_series,
            change_points=change_points,
            n_observations=len(aligned),
            start_date=aligned.index[0],
            end_date=aligned.index[-1],
        )
