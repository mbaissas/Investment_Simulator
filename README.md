# 📈 Investment Simulator

> A Python tool that uses **historical ETF data** and **Monte Carlo simulation** to project the future value of a multi-envelope investment portfolio as a range of outcomes, plus a **Machine Learning backtest** that checks whether ML can beat a naive forecast.

The simulator never produces a single "predicted" number. It builds thousands of plausible futures (**scenarios**) from real historical market months, then summarises them as ranges.

Example insights:
- *"In 20 years, you'll most likely have between 180k€ and 650k€ — median 380k€"*
- *"You have a 72% chance of reaching 100k€ within 15 years"*
- *"At the median, you'll reach 60k€ in 8.4 years"*

Domain vocabulary (Portfolio, Envelope, Holding, Asset, Contribution, Transfer…) is defined in [`GLOSSARY.md`](GLOSSARY.md). Key design decisions are recorded in [`docs/adr/`](docs/adr/).

---

## 🗂️ Project Structure

```
investment-simulator/
│
├── data/
│   └── fetcher.py          # Downloads monthly prices via yfinance, converts to EUR (with local cache)
│
├── models/
│   ├── return_model.py     # Market assets (joint block bootstrap) and fixed-rate assets
│   ├── monte_carlo.py      # Monte Carlo engine: builds scenarios month by month
│   └── backtest.py         # ML backtest: Linear Regression / Random Forest vs naive mean
│
├── simulation/
│   ├── cashflow.py         # Contribution phases: income, savings rate, allocation, transfers
│   └── simulator.py        # Full pipeline orchestrator
│
├── analysis/
│   └── metrics.py          # Percentiles, probability of reaching a target, time-to-target
│
├── visualization/
│   └── plots.py            # Matplotlib charts: projection bands, histograms
│
├── config/
│   └── portfolio_example.json  # Your portfolio configuration (edit this!)
│
├── docs/adr/               # Architecture decision records
├── GLOSSARY.md             # Domain vocabulary
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

Edit `config/portfolio_example.json`. It describes four things:

- **Assets**: what each asset's returns are based on (a ticker, a proxy, or a fixed rate), and its annual fees (TER)
- **Envelopes**: your accounts (PEA, CTO, Livret A…) and the current value of each holding
- **Contribution phases**: your money timeline, with income, savings rate, allocation and transfers, each starting at a calendar month
- **Inflation rate**: used only when displaying results in today's euros

```json
{
  "name": "My Portfolio",
  "inflation_rate": 0.02,

  "assets": {
    "MSCI World":   { "ticker": "IWDA.AS", "ter": 0.0020 },
    "Bourso Monde": { "proxy": "MSCI World", "ter": 0.0050 },
    "Livret A":     { "fixed_rate": 0.017 }
  },

  "envelopes": [
    { "name": "PEA",           "holdings": [{ "asset": "MSCI World",   "current_value": 702 }] },
    { "name": "Assurance Vie", "holdings": [{ "asset": "Bourso Monde", "current_value": 149 }] },
    { "name": "Livret A",      "holdings": [{ "asset": "Livret A",     "current_value": 13621 }] }
  ],

  "contribution_phases": [
    {
      "name": "Studies",
      "start": "2026-10",
      "income": 0,
      "transfer": { "from": "Livret A", "amount": 300 },
      "allocation": { "PEA": { "MSCI World": 70 }, "Assurance Vie": { "Bourso Monde": 30 } }
    },
    {
      "name": "Internship",
      "start": "2027-02",
      "income": 1500,
      "savings_rate": 0.20,
      "allocation": { "PEA": { "MSCI World": 70 }, "Assurance Vie": { "Bourso Monde": 30 } }
    }
  ]
}
```

A few rules:
- **Income** is your monthly *net* pay. The simulator does not convert gross to net.
- Each month, `income × savings_rate` (a **contribution**, new money) and any **transfer** (money moved out of another holding, typically the Livret A) are split across holdings according to the **allocation**. Allocation percentages must sum to 100.
- A transfer stops once its source holding is empty, and the simulator reports the month this happens.
- Each phase runs until the next one starts. The last phase runs until the horizon.

### 4. Run the simulation

```bash
# Basic run (20 years, 10,000 scenarios)
python main.py

# Custom horizon and target
python main.py --years 30 --scenarios 10000 --target 200000

# Show results in today's euros (inflation-adjusted)
python main.py --real

# Save charts to disk
python main.py --save-plots outputs/

# Headless (no window)
python main.py --no-plot

# Run the ML backtest
python main.py --backtest
```

---

## 📊 What the Charts Show

### Projection chart
Three percentile bands show the spread of outcomes:
- 🔴 **Pessimistic (p5)**: 95% of scenarios do better than this
- 🔵 **Median (p50)**: Half of scenarios are above, half below
- 🟢 **Optimistic (p95)**: Only 5% of scenarios do better than this

The yellow dashed line shows the money you put in yourself (starting values plus contributions, no returns) for reference.

### Histogram
Distribution of final portfolio values across all scenarios at a given year. It shows the "shape" of the uncertainty.

---

## 🧠 How It Works

### Step 1: Fetch historical data
`yfinance` downloads monthly closing prices for each market asset. Prices quoted in USD are converted to EUR, so returns include the currency risk a euro investor actually carries. Results are cached locally so you're not rate-limited.

### Step 2: Build the historical window
The simulator keeps the **historical window**: the months for which *every* market asset has data. It is bounded by the youngest asset (Bitcoin, from about 2014), and the window used is always displayed with the results.

### Step 3: Monte Carlo simulation
Each of the 10,000 scenarios is built by drawing random **12-month blocks** of real historical months and applying the **same months to every market asset** ([ADR 0001](docs/adr/0001-joint-block-bootstrap-returns.md)). This keeps two things real:
- **Correlation**: a crash month hits MSCI World, the S&P 500 and Bitcoin together
- **Sequences**: bad months tend to come in runs, as they do in real crises

Every month, in every scenario:
1. Market assets move by that month's historical return, minus fees (TER)
2. Fixed-rate assets grow at their fixed rate
3. Contributions and transfers from the current contribution phase are added

### Step 4: Analysis
From all scenarios, the simulator computes percentiles (p5 / p50 / p95), the probability of reaching your target, and the median time to reach it. Results are in current euros by default, or in today's euros with `--real`.

### ML backtest (separate from the projection)
Predicting next month's market return is something the whole finance industry attempts without reliable success. Markets behave close to a random walk, and ~150 monthly data points are far too few for complex models. So ML is **not** used to generate the projection.

Instead, the backtest asks an honest question: *do Linear Regression and Random Forest predict next month's return better than the naive historical mean?* Models are trained on past data and tested on later months they have never seen, moving forward one month at a time. The expected answer is "no, or barely", which is exactly why the projection relies on Monte Carlo.

---

## ⚙️ Supported Assets

Every asset is one of two kinds:
- **Market asset**: its returns come from historical prices, either its own ticker or a **proxy's** when it has none
- **Fixed-rate asset**: it grows at a fixed annual rate you set in the config, with no randomness

| Asset name | Kind | Source | Notes |
|---|---|---|---|
| MSCI World | Market | IWDA.AS | iShares Core MSCI World |
| CAC 40 | Market | ^FCHI | French index |
| S&P 500 | Market | SPY | US large cap (USD → EUR) |
| Core S&P 500 | Market | CSPX.L | iShares Core S&P 500 |
| S&P 500 Info Tech | Market | IYW | US tech sector (USD → EUR) |
| MSCI ACWI | Market | ACWI | Global equities (USD → EUR) |
| Bitcoin | Market | BTC-USD | Cryptocurrency (USD → EUR) |
| Bourso Monde | Market | proxy: MSCI World | No usable history of its own |
| SpaceX | Market | proxy: ITA | Private company, aerospace ETF used as proxy |
| Fonds Euro | Fixed rate | config | Capital-guaranteed |
| Livret A | Fixed rate | config | Regulated savings account |
| Livret Jeune | Fixed rate | config | Regulated savings account |

To add a market asset, give it a `ticker` (or a `proxy`) in the config. Choose tickers with a long history: a young ticker shrinks the historical window for **every** asset.

---

## ⚠️ Disclaimer

This tool is for **educational and personal planning purposes only**. It does not constitute financial advice. Past returns do not guarantee future performance. Known limits:

- **No taxes**: results are before tax (PFU 30% on CTO, 17.2% social contributions on fonds euro, PEA exempt from income tax after 5 years…)
- **Short history**: the historical window starts around 2014. It excludes the 2008 crisis and includes Bitcoin's exceptional run, so projections may be optimistic
- **Nominal by default**: inflation is only removed when you ask for results in today's euros, using a fixed assumed rate
- **No account limits or rebalancing**: the Livret A ceiling is not enforced, and holdings only change through returns, contributions and transfers
- **Assumed rates**: fixed-rate assets use the rates you enter, which change over time in reality

Always consult a licensed financial advisor before making investment decisions.

---

## 🛣️ Roadmap

- [ ] **Phase 1**: Core Python simulator with Monte Carlo projection and ML backtest *(current)*
- [ ] **Phase 2**: Streamlit web interface (interactive sliders, live charts)
- [ ] **Phase 3**: Regime detection with Hidden Markov Models (bull/bear market awareness)
- [ ] **Phase 4**: FastAPI backend + React frontend
