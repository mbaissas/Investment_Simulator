"""
models/return_model.py
-----------------------
Fits statistical return models to historical data.

Two approaches are supported:
  1. Parametric  — fit a normal (Gaussian) distribution to log-returns.
  2. Bootstrap   — resample blocks of actual historical returns (better captures
                   autocorrelation and fat tails).

The bootstrap method is the default and more honest approach.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


@dataclass
class ReturnModel:
    """
    A fitted return model for a single asset.

    Attributes
    ----------
    asset_name : str
        Human-readable name of the asset.
    method : str
        'parametric' or 'bootstrap'.
    annual_mean : float
        Annualized expected return (arithmetic, from log-returns).
    annual_vol : float
        Annualized volatility (standard deviation of log-returns).
    monthly_log_returns : np.ndarray
        Raw monthly log-returns used to fit the model (for bootstrap).
    """
    asset_name: str
    method: str
    annual_mean: float
    annual_vol: float
    monthly_log_returns: np.ndarray

    def __repr__(self):
        return (
            f"ReturnModel('{self.asset_name}', method='{self.method}', "
            f"E[r]={self.annual_mean:.1%}/yr, σ={self.annual_vol:.1%}/yr)"
        )

    def sample_monthly(
        self,
        n_months: int,
        n_paths: int = 1,
        rng: np.random.Generator | None = None,
        block_size: int = 6,
    ) -> np.ndarray:
        """
        Sample n_months of monthly log-returns for n_paths simulated trajectories.

        Parameters
        ----------
        n_months : int
            Number of months to simulate.
        n_paths : int
            Number of independent paths (for Monte Carlo).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility).
        block_size : int
            Block length for bootstrap resampling (default 6 months).

        Returns
        -------
        np.ndarray of shape (n_paths, n_months)
            Monthly log-returns.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.method == "parametric":
            monthly_mean = self.annual_mean / 12
            monthly_std = self.annual_vol / np.sqrt(12)
            return rng.normal(monthly_mean, monthly_std, size=(n_paths, n_months))

        elif self.method == "bootstrap":
            return _block_bootstrap(
                self.monthly_log_returns,
                n_months=n_months,
                n_paths=n_paths,
                block_size=block_size,
                rng=rng,
            )

        else:
            raise ValueError(f"Unknown method '{self.method}'")


def fit(
    returns: pd.Series | np.ndarray,
    asset_name: str = "Unknown",
    method: Literal["parametric", "bootstrap"] = "bootstrap",
) -> ReturnModel:
    """
    Fit a ReturnModel from historical monthly log-returns.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Monthly log-returns.
    asset_name : str
        Label for the asset.
    method : 'parametric' | 'bootstrap'
        Fitting method. Bootstrap is recommended.

    Returns
    -------
    ReturnModel
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 12:
        raise ValueError(
            f"Need at least 12 monthly observations to fit a model for '{asset_name}' "
            f"(got {len(r)})."
        )

    # Annualize statistics
    annual_mean = float(r.mean() * 12)
    annual_vol = float(r.std(ddof=1) * np.sqrt(12))

    print(
        f"[model] {asset_name:30s}  E[r]={annual_mean:+.1%}/yr  "
        f"σ={annual_vol:.1%}/yr  n={len(r)} months  method={method}"
    )

    return ReturnModel(
        asset_name=asset_name,
        method=method,
        annual_mean=annual_mean,
        annual_vol=annual_vol,
        monthly_log_returns=r,
    )


def _block_bootstrap(
    returns: np.ndarray,
    n_months: int,
    n_paths: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Circular block bootstrap: draw random blocks from the historical
    return series, respecting short-range autocorrelation.
    """
    n_obs = len(returns)
    n_blocks = int(np.ceil(n_months / block_size))
    result = np.empty((n_paths, n_blocks * block_size))

    for p in range(n_paths):
        starts = rng.integers(0, n_obs, size=n_blocks)
        for i, s in enumerate(starts):
            # Wrap around the series (circular)
            idx = np.arange(s, s + block_size) % n_obs
            result[p, i * block_size:(i + 1) * block_size] = returns[idx]

    return result[:, :n_months]


def fit_fallback(
    annual_mean: float,
    annual_vol: float,
    asset_name: str = "Unknown",
    n_synthetic_months: int = 240,
    seed: int = 42,
) -> ReturnModel:
    """
    Create a model without historical data, using assumed mean and volatility.
    Useful for assets without a reliable ticker (e.g. SpaceX private shares).

    Parameters
    ----------
    annual_mean : float
        Assumed annualized mean return (e.g. 0.10 for 10%).
    annual_vol : float
        Assumed annualized volatility (e.g. 0.30 for 30%).
    n_synthetic_months : int
        Number of synthetic observations to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ReturnModel
    """
    rng = np.random.default_rng(seed)
    monthly_mean = annual_mean / 12
    monthly_std = annual_vol / np.sqrt(12)
    synthetic = rng.normal(monthly_mean, monthly_std, size=n_synthetic_months)

    print(
        f"[model] {asset_name:30s}  E[r]={annual_mean:+.1%}/yr  "
        f"σ={annual_vol:.1%}/yr  (assumed — no historical data)"
    )

    return ReturnModel(
        asset_name=asset_name,
        method="bootstrap",
        annual_mean=annual_mean,
        annual_vol=annual_vol,
        monthly_log_returns=synthetic,
    )


if __name__ == "__main__":
    # Demo: fit a model on synthetic data
    rng = np.random.default_rng(0)
    fake_returns = rng.normal(0.007, 0.04, size=240)  # ~8%/yr, 14% vol
    model = fit(fake_returns, asset_name="Demo Asset", method="bootstrap")
    print(model)
    paths = model.sample_monthly(n_months=240, n_paths=5, rng=rng)
    print(f"Paths shape: {paths.shape}")
    print(f"Cumulative return (path 0): {np.expm1(paths[0].sum()):.1%}")
