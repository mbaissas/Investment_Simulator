# Investment Simulator

Projects the future value of a personal, multi-envelope portfolio as a range of plausible outcomes rather than a single number.

## Language

### Portfolio structure

**Portfolio**:
Everything the investor owns across all envelopes, at a given point in time.
_Avoid_: Patrimoine, wealth, account

**Envelope**:
A named account that groups holdings (PEA, CTO, Assurance Vie, Livret A, a crypto exchange account). Tax treatment is a property of an envelope, not what defines it.
_Avoid_: Wrapper, account, compte

**Asset**:
Something whose returns are modelled (MSCI World, Bitcoin, Livret A rate). An asset exists once, however many envelopes hold it.
_Avoid_: Stock, action, ETF, fund

**Holding**:
A given asset held inside a given envelope, with a current value and contributions.
_Avoid_: Position, line, investment

**Market Asset**:
An asset whose returns come from historical market prices, either its own or those of a proxy.
_Avoid_: Listed asset, ticker asset

**Proxy**:
A market asset whose price history stands in for an asset that has none of its own (e.g. MSCI World for Bourso Monde).
_Avoid_: Substitute, stand-in

**Fixed-Rate Asset**:
An asset that grows at a fixed, assumed annual rate with no randomness (Livret A, Livret Jeune, Fonds Euro).
_Avoid_: Safe asset, guaranteed asset, deterministic asset

### Money flows

**Contribution**:
New money entering the portfolio from outside (e.g. salary) into a holding. It increases the portfolio's total.
_Avoid_: Deposit, versement, investment, DCA

**Transfer**:
Money moved from one holding to another (e.g. Livret A → PEA). It leaves the portfolio's total unchanged. A transfer stops once its source holding is empty.
_Avoid_: Contribution, virement, rebalancing

**Contribution Phase**:
A period starting at a given calendar month with its own income, savings rate, allocation and transfers. Phases follow one another in order; the last one runs until the horizon.
_Avoid_: Salary period, regime, "after salary"

**Income**:
The investor's monthly net pay during a contribution phase, before income tax is withheld. It may be zero.
_Avoid_: Salary, gross pay, revenue

**Savings Rate**:
The share of income that becomes contributions each month.
_Avoid_: Investment rate, 20% rule

**Allocation**:
How the money invested each month is split across holdings, as percentages that sum to 100%.
_Avoid_: Weights, split, distribution

### Projection

**Scenario**:
One simulated future of the whole portfolio, month by month. Every asset follows a single return path within a given scenario.
_Avoid_: Path, run, simulation, trajectory

**Projection**:
The set of all scenarios for a given portfolio and horizon, summarised as ranges of outcomes.
_Avoid_: Forecast, prediction

**Historical Window**:
The period of past months that scenarios are drawn from, common to every market asset.
_Avoid_: Lookback, history, sample period

**Target**:
An amount of money whose probability and time of being reached are reported by a projection.
_Avoid_: Goal, objective

**Real Value**:
An amount expressed in today's euros, after removing assumed inflation. The opposite is nominal value.
_Avoid_: Inflation-adjusted, deflated

### Model evaluation

**Backtest**:
A check of whether a predictive model beats the naive historical mean at predicting next month's return, on past data it was not trained on. A backtest never feeds a projection.
_Avoid_: ML prediction, validation, forecast
