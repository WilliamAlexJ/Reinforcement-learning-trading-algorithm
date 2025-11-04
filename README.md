# Reinforcement-learning-trading-algorithm

This project implements a reinforcement learning (RL) trading system on the SPY ETF (S&P 500), designed to be both return-seeking and risk-aware. The core idea is to let an RL agent learn when to be long, flat, or short SPY, but only when there is statistical evidence that the signal is reliable and the market regime is favorable.

The system is intentionally modular:

Configuration layer(`Config`): centralizes paths, dates, hyperparameters, and risk settings.
Data & feature layer (`data_io.py`, `preprocess.py`): handles raw data and turns it into a clean feature matrix.
Statistical evidence layer (`stats_evidence.py`): evaluates whether the signal is currently trustworthy and whether the return distribution has shifted.
RL environment layer (`env_trading.py`): simulates trading with position, equity, drawdown, transaction costs, and a risk-adjusted reward.
RL agent layer (`agent_rl.py`): wraps Stable-Baselines3 (DQN / PPO) and handles training, acting, saving, and loading.
Training and backtesting entrypoints (`main_train.py`, `main_backtest.py`, `backtest.py`): orchestrate the full pipeline end-to-end.
Visualization & reporting** (`plot_results_enhanced.py`): produces an enriched equity curve with trade and regime annotations.

Backtesting on SPY using this pipeline yields:

* Total return**: 131.32%
* CAGR: 15.06%
* Annualized volatility: 14.31%
* Sharpe ratio: 1.05
* Max drawdown: −21.87%
* Hit rate (profitable days): 54.78%
* Number of periods: 1506 trading days

The rest of this report explains **what the system does, step by step, and why each step is there.

---

## 2. Configuration and Design Philosophy

All important parameters live in a single configuration dataclass (`Config`), which:

* Defines data paths (raw data and processed features).
* Sets the symbol (default: `SPY`, but easily changeable).
* Specifies train and test periods (e.g., 2010–2017 for training, 2018–2024 for testing).
* Controls the rolling window length for statistical tests (e.g., ~252 days).
* Sets risk and transaction assumptions:

  * Risk penalty weight (`risk_lambda`)
  * Transaction costs in basis points (`transaction_cost_bps`)
  * Maximum absolute position (`max_position`, e.g. −1 to +1).
* Provides basic RL settings:

  * Algorithm choice (`rl_algo`, e.g. `"dqn"`).
  * Training steps (`total_timesteps`, e.g. 50,000).
  * Discount factor (`gamma`), batch size, random seed, etc.

**Why this matters:**

* Centralizing configuration makes the system easy to reproduce and modify (e.g., changing from SPY to another ETF or adjusting training dates).
* It allows consistent use of the same assumptions across preprocessing, environment, statistical evidence, RL agent, and backtester.
* It avoids “magic numbers” scattered across files, which is vital in a research or production environment.

---

## 3. Data Acquisition and Feature Engineering

### 3.1 Data acquisition and caching

The data layer uses `yfinance` through `data_io.py` to:

1. Download raw OHLCV data (Open, High, Low, Close, Adjusted Close, Volume) for SPY over the full training + test period.
2. Cache raw data to CSV in `data/raw/SPY.csv`.
3. Reload from cache by default to avoid unnecessary API calls and ensure consistent backtests over time.

**Why this matters:**

* Ensures reproducibility, every run uses the same underlying data unless deliberately refreshed.
* Makes the system efficient by avoiding repeated downloads.
* Encapsulates all I/O in one place, keeping the rest of the pipeline pure and focused on transformations.

### 3.2 Choosing a clean price series

In `preprocess.py`, the pipeline first selects a single, robust price series:

* Prefer Adjusted Close if available (captures dividends and splits).
* Fall back to Close otherwise.
* Handles possible multi-index columns and coerces non-numeric entries to NaN, which are later dropped.

**Why this matters:**

* The model should act on a consistent, tradeable price.
* Using Adjusted Close improves realism and avoids distortions from corporate actions.
* Robust handling of different data formats makes the system less fragile.

### 3.3 Feature construction

The core feature matrix is built via `build_feature_matrix`, producing a DataFrame with:

* Price & returns

  * `price`: main tradeable price.
  * `ret_1d`: 1-day log return (captures daily changes).
  * `ret_5d`: 5-day simple return (shorter-term momentum).
  * `ret_21d`: 21-day simple return (approx. 1-month trend).

* **Volatility**

  * `vol_21d`: 21-day realized volatility, annualized.
  * `garch_vol`: conditional volatility from a GARCH(1,1) model fitted on daily returns.

* **Trend & momentum**

  * `ma_10d`: 10-day moving average.
  * `ma_50d`: 50-day moving average.
  * `mom_10d`: 10-day momentum (price change over 10 days).

* **Risk feature**

  * `drawdown`: running maximum drawdown of the price series.

After feature creation, the module:

* Aligns all series on the same date index.
* Drops rows with NaNs (caused by rolling windows or coercion).
* Optionally splits into train/test periods using the dates from `Config`.

**Why these features:**

* Short and medium-term returns and momentumlet the agent detect recent trends and mean-reversion.
* Moving averages and their relative position to price encode trend information (bull vs bear conditions).
* Realized vol and GARCH vol give a sense of current and conditional risk in the market.
* Drawdown quantifies downside risk and investor pain, which is crucial for risk-aware decisions.
* The chosen set is small but interpretable, making the behavior of the agent easier to explain.

---

## 4. Statistical Evidence Layer

The statistical evidence layer is implemented in `stats_evidence.py` via an `EvidenceEngine`. This is one of the unique parts of the system: it explicitly tells the RL agent how trustworthy the signal is and what regime the market is in.

### 4.1 Configuration

`EvidenceConfig` is built from the global `Config` and includes:

* Rolling window length (e.g. 252 days).
* Minimum samples before tests are run.
* Threshold for evidence strength.
* Threshold for KS test p-value.
* The signal column to test (here: `mom_10d`).

### 4.2 ML-based signal test

For every time step, the engine looks back over the rolling window and:

1. Takes the chosen signal (e.g. 10-day momentum) and daily returns.
2. Fits two simple Random Forest regressions:

   * A null model using no signal (just a constant predictor).
   * An alternative model using the signal as the only predictor.
3. Compares their mean squared errors. The improvement in predictive accuracy is treated as an evidence score.

Outputs include:

* `evidence_strength`: how much the signal improves predictive performance.
* `signal_valid`: a binary flag indicating whether the improvement exceeds a pre-defined threshold.

**Why this matters:**

* Instead of assuming the signal always works, the system tests in real time whether it is currently predictive.
* Using a small ML model (Random Forest) allows for non-linear relationships without adding complexity to the main RL policy.
* `signal_valid` becomes a direct input to the RL policy and a driver of rewards.

### 4.3 Distribution shift test (KS test)

The engine also runs a Kolmogorov-Smirnov test on recent returns:

* Compares the first two-thirds of the window with the last one-third.
* If the p-value is below a threshold, `dist_shift_flag` is set to 1.

**Why this matters:**

* Detects structural breaks or regime shifts where past returns no longer resemble recent ones.
* Warns the RL agent that relying on the same behavior pattern might be dangerous.

### 4.4 Volatility regime classification

Using `garch_vol` from the feature matrix, the engine:

* Computes a volatility ratio: current GARCH vol divided by its rolling mean.
* Flags high volatility if the ratio is above a threshold (e.g. 1.5).
* Combines volatility and distribution shift into a regime type:

  * 0 = normal
  * 1 = high volatility
  * 2 = shifted + high volatility

Outputs:

* `garch_vol_ratio`
* `high_vol_flag`
* `regime_type`

**Why this matters:**

* Provides a compact representation of the current risk regime.
* Allows the RL policy to behave differently in calm vs stressed markets.

### 4.5 Evidence summary

For each step, the engine returns a dictionary including:

* `signal_valid`
* `evidence_strength`
* `dist_shift_flag`
* `ks_stat`, `ks_pvalue`
* `garch_vol_ratio`
* `high_vol_flag`
* `regime_type`
* Diagnostic MSE values for the ML test

These are fed straight into the trading environment as part of the observation.

---

## 5. Trading Environment

The trading environment is implemented in `env_trading.py` using the Gymnasium API. It is responsible for:

* Translating actions into positions.
* Applying returns, transaction costs, and risk penalties.
* Producing observations that combine market features, evidence features, and risk state.

### 5.1 State (observation) design

At each time (t), the observation vector concatenates:

1. Market features from `preprocess.build_feature_matrix`:

   * Price, multi-horizon returns, realized vol, moving averages, momentum, drawdown, GARCH vol.

2. Evidence features from `EvidenceEngine`:

   * `signal_valid`, `evidence_strength`, `dist_shift_flag`,
   * `garch_vol_ratio`, `high_vol_flag`, `regime_type`.

3. Risk state:

   * Normalized current position (position divided by `max_position`).
   * Current equity drawdown (from peak equity).

The environment recomputes the evidence over a rolling window at every step before building the observation.

**Why this matters:**

* The agent sees not just raw prices but how reliable the signal is and how risky the current regime is.
* Feeding current position and drawdown gives the agent awareness of its own risk exposure, enabling learning of path-dependent behavior (like de-risking after large losses).

### 5.2 Action space and position sizing

The action space is discrete with 3 actions:

* Action 0 → short (−max_position).
* Action 1 → flat (0).
* Action 2 → long (+max_position).

In addition, the environment scales the effective maximum position based on volatility:

* Higher volatility → smaller effective max position.
* Lower volatility → agent can utilize the full max position.

**Why this matters:**

* A simple discrete scheme keeps the learning problem manageable.
* Volatility scaling allows the system to reduce risk automatically in turbulent periods without hard-coding specific rules.

### 5.3 Reward function and risk penalties

For each step, the environment:

1. Moves from step (t) to (t+1).
2. Applies the chosen position to the next-day return to compute portfolio return.
3. Subtracts transaction costs based on the change in position and a per-trade basis point cost.
4. Updates equity and computes current drawdown relative to the running peak equity.

The reward is then constructed as:

* Portfolio return
* minus a drawdown-based risk penalty, proportional to current drawdown (`risk_lambda`).
* minus regime penalties when trading in unfavourable conditions:

  * Trading while `signal_valid` is false.
  * Trading while `dist_shift_flag` indicates distribution shift.
  * Trading in high-volatility or worst regime (`regime_type` ≥ 2).

**Why this matters:**

* Encourages the agent to seek returns, but only when adjusted for risk and regime.
* Penalizes “reckless” behavior such as taking risk when the signal is weak, the distribution has shifted, or volatility is extreme.
* Aligns the RL objective more closely with what a risk manager or PM would actually care about: risk-adjusted performance, not raw return.

---

## 6. RL Agent and Training Procedure

The RL agent is implemented in `agent_rl.py` as a wrapper around Stable-Baselines3:

* Supports DQN and PPO, with the algorithm chosen via `Config`.
* Wraps the trading environment into a `DummyVecEnv` so it is compatible with Stable-Baselines3’s expected interface.
* Uses a standard MLP policy over the observation vector.
* Takes hyperparameters (discount factor, batch size, total timesteps, verbosity, seed) from `Config`.

Key capabilities:

* `train()`: runs the RL training loop for the configured number of timesteps.
* `act(state)`: returns an action (short/flat/long) for a given state, typically in deterministic mode when deployed.
* `save()`: writes the trained model to disk (e.g., `models/SPY_dqn`).
* `load()`: loads a saved model and attaches it to a fresh environment.

**Why this matters:**

* By isolating RL-specific logic into `RLAgent`, it becomes easy to:

  * Swap algorithms (e.g. DQN → PPO).
  * Use the same environment for different RL methods.
  * Plug the agent into training and backtesting scripts without duplicating boilerplate.

---

## 7. Training and Backtesting Workflow

### 7.1 Training (`main_train.py`)

The training script performs the following steps:

1. **Initialize config and seeds**

   * Load default configuration.
   * Set global seeds for `random`, `numpy`, and optionally `torch` for reproducibility.

2. **Load raw prices**

   * Use `load_raw_prices` with caching to get SPY OHLCV data.

3. **Build feature matrix**

   * Call `build_feature_matrix` to compute all technical and risk features.
   * Save the processed features to disk for later inspection and reuse.

4. **Split into train/test**

   * Use `split_train_test` to keep only the training portion for model fitting.

5. **Create EvidenceEngine**

   * Initialize `EvidenceEngine` using the global `Config`, with `mom_10d` as the signal to test.

6. **Create TradingEnv**

   * Build the training environment using the training feature matrix and the evidence engine.

7. **Initialize and train RLAgent**

   * Create the `RLAgent` with the environment and config.
   * Call `agent.train()` for `total_timesteps` (e.g., 50,000).

8. **Save model**

   * Save the trained model to disk for later backtesting and deployment.

### 7.2 Backtesting (`main_backtest.py` and `backtest.py`)

For evaluation:

1. **Load config and seeds**.
2. **Load processed features** from disk.
3. **Split into train/test** and select only the test period.
4. **Create EvidenceEngine** with the same settings as during training.
5. **Create TradingEnv** using **only test data** to avoid look-ahead bias.
6. **Load the trained RLAgent** from disk.
7. **Run backtest** using `run_backtest`:

   * Step through every date in the test set.
   * At each step, get the agent’s action and apply it in the environment.
   * Collect equity, position, actions, rewards, and decomposition (portfolio return, transaction cost, risk penalty, regime penalties).
8. **Compute performance metrics** from the equity curve and print a human-readable report.
9. **Save detailed backtest results** (equity curve and diagnostics) to CSV for plotting (`backtests/SPY_dqn_backtest_results.csv`).

---

## 8. Backtest Results and Interpretation

The reported backtest results are:

* **Total return**: 131.32%
* **CAGR**: 15.06%
* **Sharpe ratio** (annualized): 1.05
* **Annualized volatility**: 14.31%
* **Max drawdown**: −21.87%
* **Hit rate**: 54.78%
* **Number of periods**: 1506 trading days

### 8.1 Performance profile

* A CAGR of ~15% over the test period indicates strong growth relative to many passive benchmarks.
* A Sharpe ratio of ~1.0 suggests that the strategy earns roughly one unit of return for each unit of volatility risk taken. This is respectable for a single-asset daily strategy.
* Annualized volatility (~14%) is in the same ballpark as SPY’s typical volatility, but combined with a relatively high Sharpe, it implies efficient use of risk.
* Max drawdown of −21.87% indicates meaningful, but not catastrophic, downside. It is significantly less severe than some historical equity market drawdowns, suggesting that the drawdown penalties and regime awareness are doing their job.
* A hit rate of 54.78% shows that the strategy is directionally correct more often than a random coin flip, which is typical of many successful but noisy trading systems.

### 8.2 Role of the evidence and regime layer

The evidence engine and regime-aware penalties likely contribute to:

* Avoiding trades when the signal is statistically weak (`signal_valid = 0`), reducing over-trading in noisy conditions.
* De-risking during distribution shifts and high-volatility regimes via penalties and position scaling.
* Encouraging the agent to focus on high-conviction, favorable-regime periods, which improves risk-adjusted performance rather than just raw return.

### 8.3 Visual diagnostics (equity curve and trade markers)

The plotting script (`plot_results_enhanced.py`) takes the saved backtest CSV and:

* Plots the equity curve over time.
* Marks buy and sell points based on transitions in position.
* Optionally overlays regime flags, such as:

  * Vertical shading/lines when `dist_shift_flag` = 1 (distribution shift events).
  * Differences in behavior when signals are valid vs invalid.

**Why this matters:**

* Provides a visual audit trail of where the agent is active, where it stands aside, and how that aligns with market regimes.
* Allows you to quickly see if the system is behaving in a way that’s intuitively defensible, not just numerically profitable.

---

## 9. Strengths, Limitations, and Next Steps

### Strengths

* Modular architecture: clear separation of concerns (data, features, evidence, environment, RL, backtest).
* Evidence-aware RL: integrates ML-based signal validation, distribution shift detection, and volatility regime classification directly into the decision process.
* Risk-adjusted reward: explicitly penalizes drawdown and trading in bad regimes, pushing the agent toward more robust behavior.
* Reproducible and extensible: config-driven design makes it easy to adjust symbols, periods, and hyperparameters.

### Limitations and potential improvements

* Single asset: currently focused on SPY; could be extended to multi-asset portfolios.
* Fixed transaction cost model: uses a simple basis point cost; could be refined (e.g., spread, slippage models).
* Simple RL architecture: uses standard MLP policies; more advanced architectures (e.g., recurrent networks) could capture richer temporal structure.
* Single signal focus: evidence engine tests one signal (`mom_10d`); in the future, multiple signals or factor libraries could be assessed jointly.

### Possible next steps

* Extend to multiple ETFs or sectors, using the same evidence and RL framework.
* Add baseline comparisons (e.g., buy-and-hold SPY, simple momentum rules) directly into the reporting for context.
* Incorporate hyperparameter optimization for the RL agent and for evidence thresholds.
* Explore online retraining and walk-forward validation to further test robustness.

---
