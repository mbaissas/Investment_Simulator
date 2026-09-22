"""
models/monte_carlo.py
---------------------
Core Monte Carlo engine.

Given a portfolio configuration and fitted return models, simulates
`n_paths` independent wealth trajectories over a given horizon.

Each trajectory accounts for:
  - Initial capital per asset
  - Variable monthly contributions (which can change over time)
  - Compound growth using sampled log-returns
  - Portfolio-level aggregation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from models.return_model import ReturnModel


@dataclass
class AssetSimConfig:
    """Configuration for a single asset in the simulation."""
    asset_name: str
    model: ReturnModel
    initial_value: float                       # Current value in €
    monthly_contribution: Callable[[int], float]  # f(month_index) → € contribution


@dataclass
class SimulationResult:
    """
    Output of a Monte Carlo simulation.

    Attributes
    ----------
    wealth_paths : np.ndarray, shape (n_paths, n_months + 1)
        Total portfolio value at each month, for each path.
        Column 0 is the initial wealth.
    percentiles : dict
        Pre-computed percentiles: {5, 25, 50, 75, 95} → array of length n_months+1.
    n_paths : int
    n_months : int
    """
    wealth_paths: np.ndarray
    percentiles: dict
    n_paths: int
    n_months: int

    @property
    def median(self) -> np.ndarray:
        return self.percentiles[50]

    @property
    def pessimistic(self) -> np.ndarray:
        """5th percentile — only 5% of scenarios are worse."""
        return self.percentiles[5]

    @property
    def optimistic(self) -> np.ndarray:
        """95th percentile — only 5% of scenarios are better."""
        return self.percentiles[95]

    def value_at_month(self, month: int, pct: int = 50) -> float:
        """Return the portfolio value at a given month for a given percentile."""
        return float(self.percentiles[pct][month])

    def months_to_target(self, target: float, pct: int = 50) -> int | None:
        """
        Return the first month at which the portfolio reaches `target` €
        in the given percentile scenario. Returns None if never reached.
        """
        path = self.percentiles[pct]
        idx = np.argmax(path >= target)
        if path[idx] < target:
            return None
        return int(idx)

    def years_to_target(self, target: float, pct: int = 50) -> float | None:
        months = self.months_to_target(target, pct)
        if months is None:
            return None
        return months / 12


def simulate(
    assets: list[AssetSimConfig],
    n_months: int,
    n_paths: int = 10_000,
    seed: int | None = None,
    percentile_levels: list[int] = [5, 10, 25, 50, 75, 90, 95],
) -> SimulationResult:
    """
    Run a Monte Carlo simulation over all assets.

    Parameters
    ----------
    assets : list[AssetSimConfig]
        One entry per asset/holding with its model and cash flow schedule.
    n_months : int
        Simulation horizon in months.
    n_paths : int
        Number of independent scenarios.
    seed : int, optional
        Random seed for reproducibility.
    percentile_levels : list[int]
        Which percentiles to pre-compute.

    Returns
    -------
    SimulationResult
    """
    rng = np.random.default_rng(seed)

    # Total portfolio value across all paths and months
    # Shape: (n_paths, n_months + 1)
    total_wealth = np.zeros((n_paths, n_months + 1))

    # Set initial wealth (month 0)
    initial_total = sum(a.initial_value for a in assets)
    total_wealth[:, 0] = initial_total

    for asset in assets:
        # Sample return paths for this asset: shape (n_paths, n_months)
        log_returns = asset.model.sample_monthly(
            n_months=n_months,
            n_paths=n_paths,
            rng=rng,
        )

        # Simulate wealth trajectory for this asset
        # Shape: (n_paths, n_months + 1)
        asset_wealth = _simulate_asset(
            initial_value=asset.initial_value,
            log_returns=log_returns,
            contribution_schedule=asset.monthly_contribution,
            n_months=n_months,
            n_paths=n_paths,
        )

        # Add this asset's contribution to the total (starting from month 1)
        total_wealth[:, 1:] += asset_wealth[:, 1:]

    # Pre-compute percentiles
    pcts = {p: np.percentile(total_wealth, p, axis=0) for p in percentile_levels}

    return SimulationResult(
        wealth_paths=total_wealth,
        percentiles=pcts,
        n_paths=n_paths,
        n_months=n_months,
    )


def _simulate_asset(
    initial_value: float,
    log_returns: np.ndarray,
    contribution_schedule: Callable[[int], float],
    n_months: int,
    n_paths: int,
) -> np.ndarray:
    """
    Simulate wealth for a single asset across all paths.

    Parameters
    ----------
    initial_value : float
        Starting balance in €.
    log_returns : np.ndarray, shape (n_paths, n_months)
        Pre-sampled monthly log-returns.
    contribution_schedule : Callable[[int], float]
        Function of month index (0-based) → monthly contribution in €.
    n_months : int
    n_paths : int

    Returns
    -------
    np.ndarray, shape (n_paths, n_months + 1)
        Wealth at each month.
    """
    wealth = np.zeros((n_paths, n_months + 1))
    wealth[:, 0] = initial_value

    for t in range(n_months):
        contribution = contribution_schedule(t)
        growth_factor = np.exp(log_returns[:, t])          # shape (n_paths,)
        wealth[:, t + 1] = wealth[:, t] * growth_factor + contribution

    return wealth


if __name__ == "__main__":
    from models.return_model import fit_fallback

    # Quick smoke test with a single made-up asset
    model = fit_fallback(annual_mean=0.08, annual_vol=0.15, asset_name="Demo")
    asset = AssetSimConfig(
        asset_name="Demo",
        model=model,
        initial_value=10_000,
        monthly_contribution=lambda t: 200 if t >= 12 else 0,
    )

    result = simulate([asset], n_months=240, n_paths=5_000, seed=42)

    print(f"\nSimulation over 20 years ({result.n_paths:,} paths):")
    print(f"  Pessimistic (p5):  {result.pessimistic[-1]:,.0f} €")
    print(f"  Median (p50):      {result.median[-1]:,.0f} €")
    print(f"  Optimistic (p95):  {result.optimistic[-1]:,.0f} €")

    target = 60_000
    for pct in [25, 50, 75]:
        y = result.years_to_target(target, pct=pct)
        label = f"p{pct}"
        if y:
            print(f"  Reach {target:,.0f}€ ({label}): {y:.1f} years")
        else:
            print(f"  Never reaches {target:,.0f}€ ({label})")
