# main_backtest.py
"""
Backtest entrypoint for the trading RL project.

Flow:
    1. Load processed feature matrix from disk.
    2. Split into train/test (only use test here).
    3. Create EvidenceEngine (stats_evidence).
    4. Create TradingEnv on test data.
    5. Load trained RLAgent from disk.
    6. Run backtest with frozen policy (no learning).
    7. Print performance report and optionally save results.

Run:
    python main_backtest.py
"""

import random
from pathlib import Path

import numpy as np

from config import get_default_config
from data_io import load_processed_features
from preprocess import split_train_test
from stats_evidence import EvidenceEngine
from env_trading import TradingEnv
from agent_rl import RLAgent
from backtest import run_backtest, print_backtest_report


def set_global_seeds(seed: int) -> None:
    """
    Set seeds for basic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main() -> None:
    # 1. Config & seeds
    cfg = get_default_config()
    set_global_seeds(cfg.seed)

    print(f"[main_backtest] Using symbol: {cfg.symbol}")
    print(f"[main_backtest] Test period: {cfg.test_start} to {cfg.test_end}")
    print(f"[main_backtest] RL algorithm: {cfg.rl_algo.upper()}")

    
    # 2. Load processed features from disk
    print("[main_backtest] Loading processed feature matrix...")
    features_df = load_processed_features(cfg)

    # 3. Split into train/test and select test portion
    _, test_df = split_train_test(features_df, cfg)
    if test_df.empty:
        raise ValueError(
            "Test DataFrame is empty. "
            "Check your test_start/test_end dates and data availability."
        )

    print(f"[main_backtest] Test samples: {len(test_df)}")

    # 4. Create EvidenceEngine
    print("[main_backtest] Initializing EvidenceEngine...")
    evidence_engine = EvidenceEngine.from_global_config(cfg, signal_col="mom_10d")

    # 5. Create TradingEnv for test data
    print("[main_backtest] Creating TradingEnv for test data...")
    env = TradingEnv(test_df, evidence_engine, cfg)

    # 6. Load trained RLAgent
    print("[main_backtest] Loading trained RLAgent...")
    agent = RLAgent.load(env, cfg)


    # 7. Run backtest
    print("[main_backtest] Running backtest with frozen policy...")
    metrics, results_df = run_backtest(env, agent, deterministic=True)

    # 8. Print report
    print_backtest_report(metrics)

    # 9. Optionally save detailed backtest results to CSV
    save_dir = Path("backtests")
    save_dir.mkdir(parents=True, exist_ok=True)

    results_path = save_dir / f"{cfg.symbol}_{cfg.rl_algo.lower()}_backtest_results.csv"
    results_df.to_csv(results_path)
    print(f"[main_backtest] Detailed results saved to: {results_path.as_posix()}")


if __name__ == "__main__":
    main()
