"""
simulation/simulator.py
------------------------
High-level orchestrator: loads a portfolio config, fetches data,
fits return models, and runs the Monte Carlo simulation.

This is the main entry point for a full simulation run.
"""

import json
import numpy as np
from pathlib import Path

from data.fetcher import fetch_portfolio_returns
from models.return_model import fit, fit_fallback, ReturnModel
from models.monte_carlo import AssetSimConfig, SimulationResult, simulate
from simulation.cashflow import from_config

# Assets without a reliable public ticker — use assumed parameters instead
FALLBACK_PARAMS = {
    "SpaceX": {"annual_mean": 0.10, "annual_vol": 0.50},
    "Fonds Euro": {"annual_mean": 0.025, "annual_vol": 0.005},
    "Livret A": {"annual_mean": 0.030, "annual_vol": 0.001},
    "Livret Jeune": {"annual_mean": 0.035, "annual_vol": 0.001},
}


def run(
    portfolio_path: str | Path,
    horizon_years: int = 20,
    n_paths: int = 10_000,
    history_years: int = 20,
    seed: int | None = 42,
    use_cache: bool = True,
) -> SimulationResult:
    """
    Full simulation pipeline.

    Parameters
    ----------
    portfolio_path : str or Path
        Path to a portfolio JSON file (see config/portfolio_example.json).
    horizon_years : int
        How many years to project.
    n_paths : int
        Number of Monte Carlo paths.
    history_years : int
        Years of historical data to fit return models on.
    seed : int, optional
        Random seed for reproducibility.
    use_cache : bool
        Whether to use cached price data.

    Returns
    -------
    SimulationResult
    """
    print(f"\n{'='*60}")
    print(f"  Investment Simulator — Monte Carlo")
    print(f"  Horizon: {horizon_years} years | Paths: {n_paths:,}")
    print(f"{'='*60}\n")

    # 1. Load portfolio config
    with open(portfolio_path, "r", encoding="utf-8") as f:
        portfolio = json.load(f)

    salary_start_month = int(portfolio.get("salary_start_month", 12))
    n_months = horizon_years * 12

    print(f"[config] Portfolio: {portfolio['name']}")
    print(f"[config] Salary boost starts at month {salary_start_month}")

    # 2. Fetch historical data for all assets
    print("\n[data] Fetching historical returns...")
    historical = fetch_portfolio_returns(portfolio, period_years=history_years)

    # 3. Fit return models
    print("\n[model] Fitting return models...")
    models: dict[str, ReturnModel] = {}

    for envelope in portfolio["envelopes"]:
        for holding in envelope["holdings"]:
            asset = holding["asset"]
            if asset in models:
                continue  # Already fitted

            if asset in FALLBACK_PARAMS:
                params = FALLBACK_PARAMS[asset]
                models[asset] = fit_fallback(
                    annual_mean=params["annual_mean"],
                    annual_vol=params["annual_vol"],
                    asset_name=asset,
                )
            elif asset in historical:
                models[asset] = fit(historical[asset], asset_name=asset, method="bootstrap")
            else:
                print(f"[warn] No data for '{asset}', using 5%/yr assumption.")
                models[asset] = fit_fallback(0.05, 0.15, asset_name=asset)

    # 4. Build AssetSimConfig list
    print("\n[sim] Building simulation configs...")
    asset_configs: list[AssetSimConfig] = []

    for envelope in portfolio["envelopes"]:
        for holding in envelope["holdings"]:
            asset = holding["asset"]
            config = AssetSimConfig(
                asset_name=f"{envelope['name']} / {asset}",
                model=models[asset],
                initial_value=float(holding["current_value"]),
                monthly_contribution=from_config(holding, salary_start_month),
            )
            asset_configs.append(config)

            print(
                f"  + {envelope['name']:20s} | {asset:25s} | "
                f"init={holding['current_value']:>8.0f}€ | "
                f"now={holding.get('monthly_now',0):>5.0f}€/mo | "
                f"after={holding.get('monthly_after_salary', holding.get('monthly_now',0)):>5.0f}€/mo"
            )

    total_initial = sum(a.initial_value for a in asset_configs)
    print(f"\n  Total initial portfolio: {total_initial:,.0f} €")

    # 5. Run Monte Carlo
    print(f"\n[sim] Running {n_paths:,} Monte Carlo paths over {n_months} months...\n")
    result = simulate(
        assets=asset_configs,
        n_months=n_months,
        n_paths=n_paths,
        seed=seed,
    )

    # 6. Print summary
    _print_summary(result, horizon_years)

    return result


def _print_summary(result: SimulationResult, horizon_years: int):
    print(f"\n{'='*60}")
    print(f"  Results — {horizon_years}-year horizon")
    print(f"{'='*60}")

    for years in [5, 10, 20]:
        if years > horizon_years:
            continue
        month = years * 12
        p5  = result.value_at_month(month, pct=5)
        p50 = result.value_at_month(month, pct=50)
        p95 = result.value_at_month(month, pct=95)
        print(f"\n  In {years:2d} years:")
        print(f"    Pessimistic  (p5)  : {p5:>12,.0f} €")
        print(f"    Median       (p50) : {p50:>12,.0f} €")
        print(f"    Optimistic   (p95) : {p95:>12,.0f} €")

    # Time to target examples
    print()
    for target in [50_000, 100_000, 200_000, 500_000]:
        y = result.years_to_target(target, pct=50)
        if y is not None:
            print(f"  Reach {target:>8,.0f} € (median): {y:.1f} years")
        else:
            print(f"  Reach {target:>8,.0f} € (median): not reached in {horizon_years} years")


if __name__ == "__main__":
    from pathlib import Path
    result = run(
        portfolio_path=Path(__file__).parent.parent / "config" / "portfolio_example.json",
        horizon_years=20,
        n_paths=5_000,
        seed=42,
    )
