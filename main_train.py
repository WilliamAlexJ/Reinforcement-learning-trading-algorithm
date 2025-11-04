# main_train.py
"""
Training entrypoint for the trading RL project.

Flow:
    1. Load raw price data from yfinance (via data_io.load_raw_prices).
    2. Build feature matrix (via preprocess.build_feature_matrix).
    3. Restrict to training period.
    4. Create EvidenceEngine (stats_evidence).
    5. Create TradingEnv (env_trading).
    6. Create RLAgent (agent_rl) and train it.
    7. Save the trained model.

Run:
    python main_train.py
"""

import random

import numpy as np

from config import get_default_config
from data_io import load_raw_prices, save_processed_features
from preprocess import build_feature_matrix, split_train_test
from stats_evidence import EvidenceEngine
from env_trading import TradingEnv
from agent_rl import RLAgent


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
        # Torch is only needed by Stable-Baselines3; if not installed yet,
        # we just skip the torch seeding here.
        pass


def main() -> None:
    # 1. Config & seeds
    cfg = get_default_config()
    set_global_seeds(cfg.seed)

    print(f"[main_train] Using symbol: {cfg.symbol}")
    print(f"[main_train] Training period: {cfg.train_start} to {cfg.train_end}")
    print(f"[main_train] RL algorithm: {cfg.rl_algo.upper()}")
    print(f"[main_train] Total timesteps: {cfg.total_timesteps}")

    # 2. Load raw data
    print("[main_train] Loading raw price data from yfinance (with cache)...")
    raw_df = load_raw_prices(cfg, use_cache=True)

    # 3. Build feature matrix
    print("[main_train] Building feature matrix...")
    features_df = build_feature_matrix(raw_df, cfg)

    print("[main_train] Saving processed features to disk...")
    save_processed_features(features_df, cfg)

    # 4. Split into train/test and select training portion
    train_df, test_df = split_train_test(features_df, cfg)
    if train_df.empty:
        raise ValueError(
            "Training DataFrame is empty. "
            "Check your train_start/train_end dates and data availability."
        )

    print(f"[main_train] Train samples: {len(train_df)}")
    print(f"[main_train] Test samples (not used here): {len(test_df)}")

    # 5. Create EvidenceEngine
    print("[main_train] Initializing EvidenceEngine...")
    # The signal column is 'mom_10d' defined in preprocess.build_feature_matrix
    evidence_engine = EvidenceEngine.from_global_config(cfg, signal_col="mom_10d")

    # 6. Create TradingEnv
    print("[main_train] Creating TradingEnv for training data...")
    env = TradingEnv(train_df, evidence_engine, cfg)

    # 7. Create RLAgent and train
    print("[main_train] Initializing RLAgent...")
    agent = RLAgent(env, cfg)

    print("[main_train] Starting training...")
    agent.train()
    print("[main_train] Training finished.")

    # 8. Save trained model
    print("[main_train] Saving trained model...")
    model_path = agent.save()
    print(f"[main_train] Model saved to: {model_path}")


if __name__ == "__main__":
    main()
