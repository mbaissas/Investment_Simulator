# Returns come from a joint block bootstrap of history, with no parametric model

Every scenario is built by drawing random 12-month blocks of real historical months and applying each block to all market assets at once. Drawing the same months for every asset preserves cross-asset correlation for free: a crash month hits MSCI World, the S&P 500 and Bitcoin together. Keeping whole blocks preserves realistic runs of bad months. We deliberately dropped the parametric (Gaussian) alternative. It needs a covariance matrix to keep correlation, understates fat tails, and would double the code for little gain in a project meant to stay simple.

## Consequences

- The historical window is the period common to all market assets, bounded by the youngest one (Bitcoin, about 2014 onward). It excludes 2008 and includes Bitcoin's exceptional run. Projections must state the window they used.
- Adding a market asset with a short history shrinks the window for every asset.
- Machine-learning models are not a source of returns. They live in a separate backtest that checks whether they beat the naive historical mean.
