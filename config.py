# config.py
"""
Central configuration for the project.


paths, date ranges, RL hyperparameters, and risk settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    #Data paths
    data_path_raw: str = "data/raw"
    data_path_processed: str = "data/processed"


    # Later you can generalize to a list of symbols.
    symbol: str = "SPY"

    #Time ranges
    # Training / validation / test periods are defined in calendar dates.
    train_start: str = "2010-01-01"
    train_end: str = "2017-12-31"

    test_start: str = "2018-01-01"
    test_end: str = "2024-12-31"

    # Optionally, you can define a max episode length in days.
    # If None, an episode just runs from start to end of the chosen period.
    episode_length_days: Optional[int] = None

    #Statistical evidence settings
    rolling_window: int = 252             # days for rolling stats tests (~1 year)
    min_samples_for_tests: int = 126      # minimum points before running LR/KS

    lr_pvalue_threshold: float = 0.05     # significance for "signal_valid"
    ks_pvalue_threshold: float = 0.05     # significance for "dist_shift_flag"

    #Reward / risk settings
    risk_lambda: float = 5.0              # weight of risk penalty in reward
    transaction_cost_bps: float = 1.0     # per trade, in basis points
    max_position: float = 1.0             # max absolute position (e.g. -1 to +1)

    #Reward scaling (you can tweak if values are too large/small)
    reward_scale: float = 1.0

    #RL algorithm / training settings
    rl_algo: str = "dqn"                  # placeholder; used by agent_rl.py
    seed: int = 42

    total_timesteps: int = 50_000        # for RL training loops
    batch_size: int = 64
    gamma: float = 0.99                   # discount factor

    #Logging / debugging
    verbose: int = 1                      # 0 = silent, 1 = basic logs
    save_dir: str = "models"              # where trained agents/policies go


def get_default_config() -> Config:
    """
    Convenience function so other files can do:

        from config import get_default_config
        cfg = get_default_config()
    """
    return Config()
