# backtest.py
"""
Backtesting utilities for the trading project.

This module does ALL evaluation work:
- Run a frozen RL policy on a TradingEnv (no learning).
- Collect equity curve, actions, positions, rewards.
- Compute performance metrics (CAGR, Sharpe, max drawdown, hit rate).

Typical usage (from main_backtest.py):

    from config import get_default_config
    from data_io import load_processed_features
    from preprocess import split_train_test
    from stats_evidence import EvidenceEngine
    from env_trading import TradingEnv
    from agent_rl import RLAgent
    from backtest import run_backtest, print_backtest_report

    cfg = get_default_config()
    features = load_processed_features(cfg)
    train_df, test_df = split_train_test(features, cfg)

    engine = EvidenceEngine.from_global_config(cfg, signal_col="mom_10d")
    env = TradingEnv(test_df, engine, cfg)

    agent = RLAgent.load(env, cfg)   # load trained model
    metrics, results_df = run_backtest(env, agent)

    print_backtest_report(metrics)
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from env_trading import TradingEnv
from agent_rl import RLAgent


# Core backtest runner


def run_backtest(
    env: TradingEnv,
    agent: RLAgent,
    deterministic: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Run a frozen RL policy on the given environment (Gymnasium-style API).

    Parameters:
        env: TradingEnv instance initialized with test data.
        agent: RLAgent with a loaded/trained model (NO further learning here).
        deterministic: whether to use deterministic policy when available.

    Returns:
        metrics: dict of performance metrics.
        results_df: DataFrame indexed by date with columns:
            - equity
            - position
            - action
            - reward
            - portfolio_return
            - transaction_cost
            - risk_penalty
            - penalty_stats
            - drawdown
    """
    obs, info = env.reset()
    terminated = False
    truncated = False

    dates = []
    equity_curve = []
    positions = []
    actions = []
    rewards = []
    portfolio_returns = []
    transaction_costs = []
    risk_penalties = []
    penalty_stats_list = []
    drawdowns = []

    while not (terminated or truncated):
        # obs is already a NumPy array from env.reset/env.step
        action = agent.act(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        # Collect per-step info
        dates.append(info["date"])
        equity_curve.append(info["equity"])
        positions.append(info["position"])
        actions.append(action)
        rewards.append(reward)
        portfolio_returns.append(info["portfolio_return"])
        transaction_costs.append(info["transaction_cost"])
        risk_penalties.append(info["risk_penalty"])
        penalty_stats_list.append(info["penalty_stats"])
        drawdowns.append(info["drawdown"])

    results_df = pd.DataFrame(
        {
            "equity": equity_curve,
            "position": positions,
            "action": actions,
            "reward": rewards,
            "portfolio_return": portfolio_returns,
            "transaction_cost": transaction_costs,
            "risk_penalty": risk_penalties,
            "penalty_stats": penalty_stats_list,
            "drawdown": drawdowns,
        },
        index=pd.to_datetime(dates),
    )
    results_df.index.name = "date"

    metrics = compute_performance_metrics(results_df["equity"])

    return metrics, results_df


# Performance metrics


def compute_performance_metrics(equity: pd.Series) -> Dict[str, float]:
    """
    Compute basic performance metrics from an equity curve.

    Assumes:
        - equity is indexed by trading dates, ascending.
        - equity values are positive and represent cumulative equity.

    Metrics:
        - cagr: compounded annual growth rate
        - sharpe: annualized Sharpe ratio (using daily returns)
        - max_drawdown: minimum drawdown (negative number)
        - vol_annual: annualized volatility of daily returns
        - hit_rate: fraction of positive daily returns
        - total_return: final / initial - 1
        - num_days: number of periods in the series
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "vol_annual": 0.0,
            "hit_rate": 0.0,
            "total_return": 0.0,
            "num_days": float(len(equity)),
        }

    # Daily returns
    ret = equity.pct_change().dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    num_periods = len(ret)
    trading_days_per_year = 252.0

    # CAGR
    years = num_periods / trading_days_per_year
    if years > 0:
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    # Volatility (annualized)
    vol_daily = float(ret.std())
    vol_annual = float(vol_daily * np.sqrt(trading_days_per_year))

    # Sharpe ratio (assuming risk-free ~ 0)
    if vol_daily > 0:
        sharpe = float(ret.mean() / vol_daily * np.sqrt(trading_days_per_year))
    else:
        sharpe = 0.0

    # Max drawdown
    max_dd = float(compute_max_drawdown(equity))

    # Hit rate: fraction of positive daily returns
    hit_rate = float((ret > 0).mean())

    metrics = {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "vol_annual": vol_annual,
        "hit_rate": hit_rate,
        "total_return": total_return,
        "num_days": float(num_periods),
    }

    return metrics


def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Compute maximum drawdown (as a negative number).

        drawdown_t = equity_t / rolling_max_t - 1
        max_drawdown = min_t drawdown_t

    Returns:
        max_drawdown (e.g. -0.3 for -30%).
    """
    eq = equity.dropna().astype(float)
    if eq.empty:
        return 0.0

    rolling_max = eq.cummax()
    drawdown = eq / rolling_max - 1.0
    max_dd = drawdown.min()

    return float(max_dd)


# Pretty-printing


def print_backtest_report(metrics: Dict[str, float]) -> None:
    """
    Print a simple, human-readable summary of backtest performance.
    """
    print("\n=== Backtest Report ===")
    print(f"Total return:  {metrics['total_return'] * 100:6.2f}%")
    print(f"CAGR:          {metrics['cagr'] * 100:6.2f}%")
    print(f"Sharpe:        {metrics['sharpe']:6.2f}")
    print(f"Vol (ann.):    {metrics['vol_annual'] * 100:6.2f}%")
    print(f"Max drawdown:  {metrics['max_drawdown'] * 100:6.2f}%")
    print(f"Hit rate:      {metrics['hit_rate'] * 100:6.2f}%")
    print(f"Num periods:   {metrics['num_days']:6.0f}")
    print("=======================\n")
