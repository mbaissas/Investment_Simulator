"""
visualization/plots.py
-----------------------
Plotting functions for simulation results.

Uses matplotlib with a clean, dark-friendly style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

from models.monte_carlo import SimulationResult


# ── Style ──────────────────────────────────────────────────────────────────
COLORS = {
    "pessimistic": "#f87171",   # Red
    "median":      "#60a5fa",   # Blue
    "optimistic":  "#34d399",   # Green
    "band_25_75":  "#60a5fa",
    "band_5_95":   "#60a5fa",
    "contributed": "#facc15",   # Yellow
    "grid":        "#2a2a3a",
    "bg":          "#0d1117",
    "text":        "#e8ecf8",
}


def _fmt_eur(x, pos=None):
    """Formatter for y-axis labels."""
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M€"
    if x >= 1_000:
        return f"{x/1_000:.0f}k€"
    return f"{x:.0f}€"


def plot_projection(
    result: SimulationResult,
    title: str = "Portfolio Projection — Monte Carlo",
    horizon_years: int | None = None,
    initial_investment: float | None = None,
    monthly_now: float | None = None,
    monthly_after_salary: float | None = None,
    salary_start_month: int | None = None,
    target_amount: float | None = None,
    dark_mode: bool = True,
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Main projection chart: percentile bands + median line.

    Parameters
    ----------
    result : SimulationResult
    title : str
    horizon_years : int, optional
        If set, adds horizon milestone markers.
    initial_investment : float, optional
        Used to draw a "capital invested" reference line.
    monthly_now : float, optional
    monthly_after_salary : float, optional
    salary_start_month : int, optional
    target_amount : float, optional
        Draws a horizontal dashed line at this value.
    dark_mode : bool
        Use dark background (default True).
    figsize : tuple
    save_path : str, optional
        If set, saves the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bg = COLORS["bg"] if dark_mode else "white"
    fg = COLORS["text"] if dark_mode else "#111827"
    grid_c = COLORS["grid"] if dark_mode else "#e5e7eb"

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    months = np.arange(result.n_months + 1)
    years_axis = months / 12

    p5  = result.percentiles[5]
    p25 = result.percentiles[25]
    p50 = result.percentiles[50]
    p75 = result.percentiles[75]
    p95 = result.percentiles[95]

    # Confidence bands
    ax.fill_between(years_axis, p5, p95, alpha=0.12,
                    color=COLORS["band_5_95"], label="5th–95th percentile")
    ax.fill_between(years_axis, p25, p75, alpha=0.22,
                    color=COLORS["band_25_75"], label="25th–75th percentile")

    # Scenario lines
    ax.plot(years_axis, p5,  color=COLORS["pessimistic"], lw=1.5, ls="--", alpha=0.85, label="Pessimistic (p5)")
    ax.plot(years_axis, p50, color=COLORS["median"],      lw=2.5,          label="Median (p50)")
    ax.plot(years_axis, p95, color=COLORS["optimistic"],  lw=1.5, ls="--", alpha=0.85, label="Optimistic (p95)")

    # Salary start marker
    if salary_start_month is not None:
        salary_year = salary_start_month / 12
        ax.axvline(salary_year, color=COLORS["contributed"], lw=1.2, ls=":", alpha=0.7)
        ax.text(salary_year + 0.1, ax.get_ylim()[1] * 0.95,
                "+ salary", color=COLORS["contributed"], fontsize=9, va="top")

    # Capital invested reference
    if all(v is not None for v in [initial_investment, monthly_now, monthly_after_salary, salary_start_month]):
        invested = []
        for m in months:
            p1 = min(m, salary_start_month)
            p2 = max(0, m - salary_start_month)
            invested.append(initial_investment + p1 * monthly_now + p2 * monthly_after_salary)
        ax.plot(years_axis, invested, color=COLORS["contributed"],
                lw=1.5, ls="-.", alpha=0.6, label="Capital invested (no growth)")

    # Target line
    if target_amount is not None:
        ax.axhline(target_amount, color="white", lw=1, ls=":", alpha=0.4)
        ax.text(years_axis[-1], target_amount,
                f" {_fmt_eur(target_amount)}", color="white",
                fontsize=9, va="bottom", ha="right", alpha=0.6)

    # Milestone annotations at 5, 10, 20 years
    if horizon_years:
        for yr in [5, 10, 20, 30]:
            if yr > horizon_years:
                continue
            m = yr * 12
            val = p50[m]
            ax.plot(yr, val, "o", color=COLORS["median"], markersize=7, zorder=5)
            ax.annotate(
                f" {_fmt_eur(val)}",
                xy=(yr, val),
                fontsize=9,
                color=fg,
                va="bottom",
            )

    # Axes
    ax.set_xlabel("Years", color=fg, fontsize=11)
    ax.set_ylabel("Portfolio value", color=fg, fontsize=11)
    ax.set_title(title, color=fg, fontsize=14, fontweight="bold", pad=16)
    ax.tick_params(colors=fg)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_eur))
    ax.set_xlim(0, result.n_months / 12)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
    ax.grid(True, color=grid_c, lw=0.5, alpha=0.5)

    # Legend
    legend = ax.legend(
        loc="upper left",
        framealpha=0.15,
        labelcolor=fg,
        edgecolor=grid_c,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(fg)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=bg, edgecolor="none")
        print(f"[plot] Saved to {save_path}")

    return fig


def plot_histogram_at_year(
    result: SimulationResult,
    year: int = 20,
    dark_mode: bool = True,
    figsize: tuple = (9, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Distribution histogram of portfolio values at a given year.
    """
    bg = COLORS["bg"] if dark_mode else "white"
    fg = COLORS["text"] if dark_mode else "#111827"
    grid_c = COLORS["grid"] if dark_mode else "#e5e7eb"

    month = min(year * 12, result.n_months)
    values = result.wealth_paths[:, month] / 1_000  # in k€

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    n_bins = min(80, result.n_paths // 50)
    ax.hist(values, bins=n_bins, color=COLORS["median"], alpha=0.7, edgecolor="none")

    for pct, color, label in [
        (5,  COLORS["pessimistic"], "p5"),
        (50, COLORS["median"],      "p50"),
        (95, COLORS["optimistic"],  "p95"),
    ]:
        v = np.percentile(values, pct)
        ax.axvline(v, color=color, lw=2, ls="--")
        ax.text(v, ax.get_ylim()[1] * 0.9, f" {label}\n {v:.0f}k€",
                color=color, fontsize=9, va="top")

    ax.set_xlabel("Portfolio value (k€)", color=fg, fontsize=11)
    ax.set_ylabel("Number of scenarios", color=fg, fontsize=11)
    ax.set_title(f"Distribution of outcomes at year {year}", color=fg, fontsize=13, fontweight="bold")
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(grid_c)
    ax.grid(True, color=grid_c, lw=0.5, alpha=0.4, axis="y")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=bg, edgecolor="none")
        print(f"[plot] Saved to {save_path}")

    return fig


if __name__ == "__main__":
    from models.return_model import fit_fallback
    from models.monte_carlo import AssetSimConfig, simulate
    from simulation.cashflow import step

    model = fit_fallback(0.08, 0.15, "Demo")
    asset = AssetSimConfig("Demo", model, 20_000, step(300, 800, 12))
    result = simulate([asset], n_months=240, n_paths=3_000, seed=0)

    fig1 = plot_projection(result, "Demo Projection", horizon_years=20,
                           initial_investment=20_000, monthly_now=300,
                           monthly_after_salary=800, salary_start_month=12)
    fig2 = plot_histogram_at_year(result, year=20)
    plt.show()
