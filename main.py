"""
main.py
-------
Entry point for the Investment Simulator.

Usage:
    python main.py
    python main.py --portfolio config/portfolio_example.json --years 30 --paths 10000
    python main.py --target 100000 --years 20
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from simulation.simulator import run
from analysis.metrics import summary_table, print_summary_table, compute_probability_above
from visualization.plots import plot_projection, plot_histogram_at_year


DEFAULT_PORTFOLIO = Path(__file__).parent / "config" / "portfolio_example.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo investment simulator with historical ETF data."
    )
    parser.add_argument(
        "--portfolio", type=Path, default=DEFAULT_PORTFOLIO,
        help="Path to portfolio JSON config file."
    )
    parser.add_argument(
        "--years", type=int, default=20,
        help="Simulation horizon in years (default: 20)."
    )
    parser.add_argument(
        "--paths", type=int, default=10_000,
        help="Number of Monte Carlo paths (default: 10000)."
    )
    parser.add_argument(
        "--target", type=float, default=None,
        help="Optional target amount in € to compute time-to-reach."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting (useful for headless/CI environments)."
    )
    parser.add_argument(
        "--save-plots", type=Path, default=None,
        help="Directory to save plots as PNG files."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Run simulation ──────────────────────────────────────────────────────
    result = run(
        portfolio_path=args.portfolio,
        horizon_years=args.years,
        n_paths=args.paths,
        seed=args.seed,
    )

    # Load portfolio for metadata
    import json
    with open(args.portfolio, "r", encoding="utf-8") as f:
        portfolio = json.load(f)

    initial = sum(
        h["current_value"]
        for e in portfolio["envelopes"]
        for h in e["holdings"]
    )
    monthly_now = portfolio.get("total_monthly_now", 300)
    monthly_after = portfolio.get("total_monthly_after_salary", 862)
    salary_month = portfolio.get("salary_start_month", 12)

    # ── Summary table ───────────────────────────────────────────────────────
    metrics = summary_table(
        result,
        initial_investment=initial,
        monthly_now=monthly_now,
        monthly_after_salary=monthly_after,
        salary_start_month=salary_month,
        horizons=[5, 10, 20, 30],
    )
    print_summary_table(metrics)

    # ── Target analysis ─────────────────────────────────────────────────────
    if args.target:
        print(f"\n{'='*60}")
        print(f"  Analysis: reaching {args.target:,.0f} €")
        print(f"{'='*60}")
        for pct in [25, 50, 75]:
            y = result.years_to_target(args.target, pct=pct)
            label = {25: "optimistic", 50: "median", 75: "pessimistic"}[pct]
            if y is not None:
                print(f"  {label:15s} (p{100-pct}): {y:.1f} years")
            else:
                print(f"  {label:15s} (p{100-pct}): not reached in {args.years} years")

        # Probability of reaching target at final horizon
        prob = compute_probability_above(result, args.target, at_month=args.years * 12)
        print(f"\n  Probability of reaching {args.target:,.0f}€ in {args.years} years: {prob:.1%}")

    # ── Plots ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        save_dir = args.save_plots
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        fig1 = plot_projection(
            result,
            title=f"Portfolio Projection — {portfolio.get('name', 'My Portfolio')}",
            horizon_years=args.years,
            initial_investment=initial,
            monthly_now=monthly_now,
            monthly_after_salary=monthly_after,
            salary_start_month=salary_month,
            target_amount=args.target,
            save_path=str(save_dir / "projection.png") if save_dir else None,
        )

        fig2 = plot_histogram_at_year(
            result,
            year=min(args.years, 20),
            save_path=str(save_dir / "histogram.png") if save_dir else None,
        )

        plt.show()


if __name__ == "__main__":
    main()
