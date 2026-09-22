"""
analysis/metrics.py
--------------------
Statistical metrics and derived insights from a SimulationResult.
"""

import numpy as np
from dataclasses import dataclass
from models.monte_carlo import SimulationResult


@dataclass
class HorizonMetrics:
    """Key metrics at a specific time horizon."""
    years: int
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    initial_investment: float
    total_contributed: float  # capital investi (dépôts)

    @property
    def real_gain_median(self) -> float:
        return self.p50 - self.total_contributed

    @property
    def multiple_median(self) -> float:
        return self.p50 / self.initial_investment if self.initial_investment > 0 else float("nan")

    @property
    def prob_above_contributed(self) -> str:
        """Not computed here — use compute_probability_above for full path data."""
        return "use compute_probability_above()"

    def __repr__(self):
        return (
            f"HorizonMetrics(t={self.years}yr | "
            f"p5={self.p5:,.0f}€  p50={self.p50:,.0f}€  p95={self.p95:,.0f}€ | "
            f"×{self.multiple_median:.1f} vs initial)"
        )


def compute_horizon_metrics(
    result: SimulationResult,
    years: int,
    initial_investment: float,
    total_contributed: float,
) -> HorizonMetrics:
    """
    Compute key metrics at a specific year horizon.

    Parameters
    ----------
    result : SimulationResult
    years : int
    initial_investment : float
        Total portfolio value at t=0.
    total_contributed : float
        All deposits made up to this horizon (initial + contributions).
    """
    month = min(years * 12, result.n_months)
    return HorizonMetrics(
        years=years,
        p5=result.value_at_month(month, 5),
        p25=result.value_at_month(month, 25),
        p50=result.value_at_month(month, 50),
        p75=result.value_at_month(month, 75),
        p95=result.value_at_month(month, 95),
        initial_investment=initial_investment,
        total_contributed=total_contributed,
    )


def compute_probability_above(
    result: SimulationResult,
    target: float,
    at_month: int,
) -> float:
    """
    Probability (across all paths) that the portfolio exceeds `target`
    at a specific month.

    Parameters
    ----------
    result : SimulationResult
    target : float
        Target portfolio value in €.
    at_month : int
        Month index.

    Returns
    -------
    float
        Probability in [0, 1].
    """
    values_at_t = result.wealth_paths[:, at_month]
    return float((values_at_t >= target).mean())


def compute_drawdown_stats(result: SimulationResult) -> dict:
    """
    Compute maximum drawdown statistics across all paths.

    Returns
    -------
    dict with keys:
        'median_max_dd': median maximum drawdown across paths
        'p95_max_dd': 95th-percentile maximum drawdown (worst 5% of paths)
    """
    max_drawdowns = []
    for path in result.wealth_paths:
        running_max = np.maximum.accumulate(path)
        dd = (path - running_max) / np.where(running_max > 0, running_max, 1)
        max_drawdowns.append(float(dd.min()))

    arr = np.array(max_drawdowns)
    return {
        "median_max_dd": float(np.percentile(arr, 50)),
        "p95_max_dd": float(np.percentile(arr, 5)),  # 5th pct = worst 5% of paths
    }


def summary_table(
    result: SimulationResult,
    initial_investment: float,
    monthly_now: float,
    monthly_after_salary: float,
    salary_start_month: int,
    horizons: list[int] = [5, 10, 20, 30],
) -> list[HorizonMetrics]:
    """
    Build a list of HorizonMetrics for multiple horizons, with estimated
    total capital contributed at each horizon.

    Parameters
    ----------
    result : SimulationResult
    initial_investment : float
    monthly_now : float
        Monthly contribution before salary.
    monthly_after_salary : float
        Monthly contribution after salary.
    salary_start_month : int
    horizons : list[int]

    Returns
    -------
    list[HorizonMetrics]
    """
    metrics = []
    for years in horizons:
        if years * 12 > result.n_months:
            continue

        months = years * 12
        phase1 = min(months, salary_start_month)
        phase2 = max(0, months - salary_start_month)
        contributed = (
            initial_investment
            + phase1 * monthly_now
            + phase2 * monthly_after_salary
        )

        m = compute_horizon_metrics(
            result, years,
            initial_investment=initial_investment,
            total_contributed=contributed,
        )
        metrics.append(m)

    return metrics


def print_summary_table(metrics: list[HorizonMetrics]):
    header = f"{'Horizon':>8} | {'p5':>10} | {'p25':>10} | {'p50 (median)':>12} | {'p75':>10} | {'p95':>10} | {'×initial':>9}"
    print("\n" + header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{str(m.years) + ' yr':>8} | "
            f"{m.p5:>10,.0f} | "
            f"{m.p25:>10,.0f} | "
            f"{m.p50:>12,.0f} | "
            f"{m.p75:>10,.0f} | "
            f"{m.p95:>10,.0f} | "
            f"×{m.multiple_median:>7.1f}"
        )
