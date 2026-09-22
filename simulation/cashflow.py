"""
simulation/cashflow.py
-----------------------
Defines contribution schedules — how much money is invested each month
per asset, potentially changing over time (e.g. when a salary starts).

A contribution schedule is a callable: f(month_index: int) -> float (€)
"""

from typing import Callable
from dataclasses import dataclass, field


@dataclass
class ContributionPhase:
    """A period of constant monthly contribution."""
    start_month: int       # Inclusive
    end_month: int | None  # Inclusive; None = infinite
    amount: float          # Monthly contribution in €

    def contains(self, month: int) -> bool:
        if month < self.start_month:
            return False
        if self.end_month is not None and month > self.end_month:
            return False
        return True


def build_schedule(phases: list[ContributionPhase]) -> Callable[[int], float]:
    """
    Build a contribution schedule function from a list of phases.
    Phases are checked in order; the first matching phase wins.
    If no phase matches, contribution is 0.

    Parameters
    ----------
    phases : list[ContributionPhase]
        Ordered list of contribution phases.

    Returns
    -------
    Callable[[int], float]
        f(month_index) → contribution in €
    """
    def schedule(month: int) -> float:
        for phase in phases:
            if phase.contains(month):
                return phase.amount
        return 0.0

    return schedule


def constant(amount: float) -> Callable[[int], float]:
    """A simple constant monthly contribution."""
    return lambda _: amount


def zero() -> Callable[[int], float]:
    """No monthly contribution (e.g. Livret A, SpaceX)."""
    return lambda _: 0.0


def step(
    before: float,
    after: float,
    change_month: int,
) -> Callable[[int], float]:
    """
    A two-phase schedule: `before` until `change_month - 1`,
    then `after` from `change_month` onwards.

    Parameters
    ----------
    before : float
        Monthly contribution before the change.
    after : float
        Monthly contribution after the change.
    change_month : int
        Month index at which the contribution changes (0-based).

    Example
    -------
    >>> # 150 €/month now, 450 €/month after 12 months (1 year)
    >>> schedule = step(before=150, after=450, change_month=12)
    """
    return build_schedule([
        ContributionPhase(0, change_month - 1, before),
        ContributionPhase(change_month, None, after),
    ])


def from_config(holding: dict, salary_start_month: int) -> Callable[[int], float]:
    """
    Build a contribution schedule from a holding config dict.

    Expected keys in `holding`:
      - 'monthly_now': float        — current monthly contribution
      - 'monthly_after_salary': float (optional) — contribution after salary
        (defaults to 'monthly_now' if not provided)

    Parameters
    ----------
    holding : dict
        A single holding entry from the portfolio config.
    salary_start_month : int
        Month index at which the salary-boosted contributions begin.

    Returns
    -------
    Callable[[int], float]
    """
    now = float(holding.get("monthly_now", 0))
    after = float(holding.get("monthly_after_salary", now))

    if now == after or salary_start_month <= 0:
        return constant(now)

    return step(before=now, after=after, change_month=salary_start_month)


if __name__ == "__main__":
    # Example: 150 €/month for 12 months, then 450 €/month
    sched = step(150, 450, change_month=12)
    for m in [0, 6, 11, 12, 24]:
        print(f"Month {m:3d}: {sched(m):.0f} €")
