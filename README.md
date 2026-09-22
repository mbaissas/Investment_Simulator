# 📈 Investment Simulator

> A Python tool that uses **historical ETF data** and **predictive modeling** (Monte Carlo simulation & Machine Learning) to project the future value of a multi-envelope investment portfolio.

This simulator provides probabilistic forecasts using multiple approaches:
- **Monte Carlo**: Runs thousands of scenarios based on real historical return distributions
- **Machine Learning**: Uses Random Forest, Linear Regression, and other models to predict portfolio evolution

Example insights:
- *"In 20 years, you'll most likely have between 180k€ and 650k€ — median 380k€"*
- *"You have a 72% chance of reaching 100k€ within 15 years"*
- *"At the median, you'll reach 60k€ in 8.4 years"*

---

## 🗂️ Project Structure

```
investment-simulator/
│
├── data/
│   └── fetcher.py          # Downloads historical prices via yfinance (with local cache)
│
├── models/
│   ├── return_model.py     # Fits return distributions to historical data
│   ├── monte_carlo.py      # Monte Carlo simulation engine
│   └── ml_predictor.py     # ML models (Random Forest, Linear Regression, etc.)
│
├── simulation/
│   ├── cashflow.py         # Variable contribution schedules (e.g. salary increase)
│   └── simulator.py        # Full pipeline orchestrator
│
├── analysis/
│   └── metrics.py          # Percentiles, time-to-target, probability analysis
│
├── visualization/
│   └── plots.py            # Matplotlib charts: projection bands, histograms
│
├── config/
│   └── portfolio_example.json  # Your portfolio configuration (edit this!)
│
└── main.py                 # Entry point — run this
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/investment-simulator.git
cd investment-simulator
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your portfolio

Edit `config/portfolio_example.json` with your actual holdings, current values, and monthly contributions:

```json
{
  "name": "My Portfolio",
  "salary_start_month": 12,
  "envelopes": [
    {
      "name": "PEA",
      "holdings": [
        {
          "asset": "MSCI World",
          "current_value": 5000,
          "monthly_now": 150,
          "monthly_after_salary": 450
        }
      ]
    }
  ]
}
```

### 4. Run the simulation

```bash
# Basic run (20 years, 10,000 paths)
python main.py

# Custom horizon and target
python main.py --years 30 --paths 10000 --target 200000

# Save charts to disk
python main.py --save-plots outputs/

# Headless (no window)
python main.py --no-plot
```

---

## 📊 What the Charts Show

### Projection chart
Three percentile bands show the spread of outcomes:
- 🔴 **Pessimistic (p5)**: 95% of scenarios do better than this
- 🔵 **Median (p50)**: Half of scenarios are above, half below
- 🟢 **Optimistic (p95)**: Only 5% of scenarios do better than this

The yellow dashed line shows pure capital invested (no returns) for reference.

### Histogram
Distribution of final portfolio values across all paths at a given year — lets you see the "shape" of uncertainty.

---

## 🧠 How It Works

### Step 1: Fetch historical data
`yfinance` downloads monthly closing prices for each ETF over 20 years. Results are cached locally so you're not rate-limited.

### Step 2: Fit return models
For each asset, the simulator estimates the **statistical distribution** of monthly log-returns from historical data. Two methods are available:

- **Bootstrap** *(default)*: resamples blocks of actual historical returns, preserving autocorrelation and fat tails
- **Parametric**: fits a Gaussian (normal) distribution — simpler but assumes returns are i.i.d.

### Step 3: Monte Carlo simulation
For each of the 10,000 independent paths, the simulator:
1. Draws random monthly returns from the fitted model
2. Applies compound growth month by month
3. Adds monthly contributions (which can change — e.g. when a salary starts)
4. Aggregates across all assets

### Step 4: Analysis
From the 10,000 final values, the simulator computes percentiles, time-to-target, and probability of reaching your goal.

---

## ⚙️ Supported Assets

The simulator maps friendly names to Yahoo Finance tickers:

| Asset name | Ticker | Notes |
|---|---|---|
| MSCI World | IWDA.AS | iShares Core MSCI World |
| CAC 40 | ^FCHI | French index |
| S&P 500 | SPY | US large cap |
| Core S&P 500 | CSPX.L | iShares Core S&P 500 |
| S&P 500 Info Tech | IYW | US tech sector |
| MSCI ACWI | ACWI | Global equities |
| Bitcoin | BTC-USD | Cryptocurrency |
| SpaceX | *(assumed)* | Private — no ticker |
| Fonds Euro | *(assumed)* | Capital-guaranteed |
| Livret A | *(assumed)* | Regulated account |

To add a new asset, add an entry to `TICKER_MAP` in `data/fetcher.py`.

---

## ⚠️ Disclaimer

This tool is for **educational and personal planning purposes only**. It does not constitute financial advice. Past returns do not guarantee future performance. The simulation does not account for:

- Taxes at withdrawal (PFU 30% on CTO, 17.2% PS on fonds euro, PEA exempt after 5 years)
- Inflation (~2%/year erodes real purchasing power)
- Management fees (TER of ETFs)
- Extreme tail events (2008-level crashes, etc.)

Always consult a licensed financial advisor before making investment decisions.

---

## 🛣️ Roadmap

- [ ] **Phase 1** — Core Python simulator with MC & ML models *(current)*
- [ ] **Phase 2** — Streamlit web interface (interactive sliders, live charts)
- [ ] **Phase 3** — Regime detection with Hidden Markov Models (bull/bear market awareness)
- [ ] **Phase 4** — FastAPI backend + React frontend
