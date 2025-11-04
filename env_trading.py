# env_trading.py
"""
Gymnasium trading environment for the project.

This environment:
- Uses precomputed features (from preprocess.build_feature_matrix)
- Calls the statistical EvidenceEngine on a rolling window each step
- Lets an RL agent choose a position: short / flat / long
- Computes *risk-adjusted* reward:
    portfolio return
    - drawdown-based penalty
    - regime / volatility-based penalties

Key ideas:
- Evidence layer (stats_evidence.EvidenceEngine) defines regime & signal validity
- RL layer sees evidence + features and learns how to act
"""

from typing import Dict, List, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from config import Config, get_default_config
from stats_evidence import EvidenceEngine


class TradingEnv(gym.Env):
    """
    Trading environment compatible with Gymnasium and Stable-Baselines3.

    Action space:
        Discrete(3):
            0 -> short  (position = -max_position, scaled by volatility regime)
            1 -> flat   (position = 0)
            2 -> long   (position = +max_position, scaled by volatility regime)

    Observation:
        1D float32 vector concatenating:
            - price/technical features at current time t
            - evidence features from statistical tests (including GARCH/regime)
            - risk state (normalized position, equity drawdown)

    Reward:
        reward_t = portfolio_return_t
                   - risk_lambda * drawdown_t
                   - small penalties when trading in bad regimes
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features_df: pd.DataFrame,
        evidence_engine: EvidenceEngine,
        cfg: Optional[Config] = None,
    ):
        super().__init__()

        self.cfg = cfg or get_default_config()
        self.features = features_df.sort_index().copy()
        self.engine = evidence_engine

        if self.features.index.name is None:
            self.features.index.name = "Date"

        # --- Which columns are used in the state ---
        self.feature_cols: List[str] = [
            "price",
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "vol_21d",
            "ma_10d",
            "ma_50d",
            "mom_10d",
            "drawdown",
            "garch_vol",
        ]

        # Evidence keys we will pull from EvidenceEngine output
        # (must match what stats_evidence.EvidenceEngine.compute_tests returns)
        self.evidence_keys: List[str] = [
            "signal_valid",
            "evidence_strength",
            "dist_shift_flag",
            "garch_vol_ratio",
            "high_vol_flag",
            "regime_type",
        ]

        # Risk state features: [normalized position, equity drawdown]
        self._risk_state_dim = 2

        for col in self.feature_cols:
            if col not in self.features.columns:
                raise ValueError(
                    f"features_df must contain column '{col}'. "
                    "Make sure preprocess.build_feature_matrix created it."
                )

        #Episode indices
        self.n_steps_total = len(self.features)
        if self.n_steps_total < 2:
            raise ValueError("Need at least 2 rows of data to run the environment.")

        self.history_window = self.cfg.rolling_window
        self.start_step_index = max(self.history_window, 1)
        self.end_step_index = self.n_steps_total - 1  # last index used as next_step

        if self.start_step_index >= self.end_step_index:
            raise ValueError(
                "Not enough data points compared to rolling_window "
                f"(rolling_window={self.history_window}, n={self.n_steps_total})."
            )

        #Gymnasium spaces
        self.action_space = spaces.Discrete(3)

        obs_dim = len(self.feature_cols) + len(self.evidence_keys) + self._risk_state_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        #Internal state (set in reset)
        self.current_step: int = 0
        self.position: float = 0.0
        self.equity: float = 1.0
        self.peak_equity: float = 1.0
        self.current_drawdown: float = 0.0
        self.last_evidence: Dict[str, float] = {}

    # Gymnasium API

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment to the beginning of a new episode.

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        self.current_step = self.start_step_index
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.current_drawdown = 0.0
        self.last_evidence = {}

        obs = self._get_observation(self.current_step)
        info: Dict = {}
        return obs, info

    def step(self, action: int):
        """
        Take one environment step.

        Parameters:
            action: integer in {0, 1, 2}.

        Returns:
            observation, reward, terminated, truncated, info
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Ensure evidence is up to date for the current step
        _ = self._get_observation(self.current_step)

        # Volatility-aware max position using GARCH vol ratio
        vol_ratio = float(self.last_evidence.get("garch_vol_ratio", 1.0))
        vol_scale = 1.0 / max(1.0, vol_ratio)  # reduce size when vol is high
        base_max_pos = self.cfg.max_position
        effective_max_pos = base_max_pos * vol_scale

        if action == 0:
            target_position = -effective_max_pos
        elif action == 1:
            target_position = 0.0
        else:
            target_position = effective_max_pos

        # Transaction costs based on position change (bps)
        position_change = target_position - self.position
        cost_per_unit = self.cfg.transaction_cost_bps * 1e-4
        transaction_cost = cost_per_unit * abs(position_change)

        # Move to next time step
        next_step = self.current_step + 1
        terminated = next_step >= self.end_step_index
        truncated = False  # no separate truncation logic here

        # Realized return at next_step (log return approx, but treat as simple)
        ret_next = float(self.features.iloc[next_step]["ret_1d"])
        portfolio_return = target_position * ret_next

        # Update equity
        self.equity *= (1.0 + portfolio_return - transaction_cost)

        # Update peak equity and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        if self.peak_equity > 0:
            self.current_drawdown = max(0.0, 1.0 - self.equity / self.peak_equity)
        else:
            self.current_drawdown = 0.0

        # Base risk penalty from drawdown
        risk_penalty = self.cfg.risk_lambda * self.current_drawdown

        # Evidence-based penalties (regime awareness)
        evidence = self.last_evidence or {}
        signal_valid = evidence.get("signal_valid", 0.0)
        dist_shift_flag = evidence.get("dist_shift_flag", 0.0)
        high_vol_flag = evidence.get("high_vol_flag", 0.0)
        regime_type = evidence.get("regime_type", 0.0)

        penalty_stats = 0.0
        if abs(target_position) > 0.0:
            # Trading when signal is statistically weak
            if signal_valid < 0.5:
                penalty_stats += 0.001
            # Trading when KS says distribution shifted
            if dist_shift_flag > 0.5:
                penalty_stats += 0.001
            # Trading in high-vol regimes
            if high_vol_flag > 0.5:
                penalty_stats += 0.001
            # Trading in worst regime (e.g., 2 = shifted + high vol)
            if regime_type >= 2.0:
                penalty_stats += 0.001

        # Final shaped reward
        reward = portfolio_return - risk_penalty - penalty_stats

        # Commit new state
        self.position = target_position
        self.current_step = next_step

        obs_next = self._get_observation(self.current_step)

        info = {
            "equity": self.equity,
            "drawdown": self.current_drawdown,
            "portfolio_return": portfolio_return,
            "transaction_cost": transaction_cost,
            "risk_penalty": risk_penalty,
            "penalty_stats": penalty_stats,
            "position": self.position,
            "date": self.features.index[self.current_step],
            "evidence": self.last_evidence,
        }

        return obs_next, float(reward), bool(terminated), bool(truncated), info

    # Internal helpers

    def _get_observation(self, step_idx: int) -> np.ndarray:
        """
        Build the observation at a given step index.

        Combines:
            - price/technical features at step_idx
            - evidence from statistical tests on rolling window up to step_idx
            - risk state (current position, equity drawdown)
        """
        row = self.features.iloc[step_idx]
        feat_vec = row[self.feature_cols].astype(float).values

        # Rolling window for evidence
        start_idx = max(0, step_idx - self.history_window + 1)
        history_window = self.features.iloc[start_idx : step_idx + 1]

        evidence = self.engine.compute_tests(history_window)
        self.last_evidence = evidence

        evidence_vec = np.array(
            [float(evidence.get(k, 0.0)) for k in self.evidence_keys],
            dtype=np.float32,
        )

        norm_pos = 0.0
        if self.cfg.max_position > 0:
            norm_pos = float(self.position / self.cfg.max_position)

        risk_vec = np.array(
            [norm_pos, self.current_drawdown],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                feat_vec.astype(np.float32),
                evidence_vec,
                risk_vec,
            ],
            axis=0,
        )

        return obs

    def render(self):
        """
        Basic text render (optional).
        """
        date = self.features.index[self.current_step]
        print(
            f"Date: {date}, "
            f"Step: {self.current_step}, "
            f"Pos: {self.position:.2f}, "
            f"Equity: {self.equity:.4f}, "
            f"Drawdown: {self.current_drawdown:.4f}"
        )

    def close(self):
        """
        Nothing special to clean up in this minimal env.
        """
        pass
