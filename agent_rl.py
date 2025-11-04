# agent_rl.py
"""
RL agent wrapper for the trading project.

This module does ALL RL-specific work:
- Wraps the trading environment in a vectorized env for Stable-Baselines3.
- Creates the RL model (DQN or PPO) based on config.
- Provides simple methods:
    - train()
    - act(state)
    - save()
    - load(...)

Dependencies:
    pip install stable-baselines3[extra] torch gym
"""

from pathlib import Path
from typing import Optional, Type

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from config import Config, get_default_config
from env_trading import TradingEnv


ALGO_MAP = {
    "dqn": DQN,
    "ppo": PPO,
}


def _make_vec_env(env: TradingEnv) -> DummyVecEnv:
    """
    Wrap a single TradingEnv instance into a DummyVecEnv for Stable-Baselines3.

    
    """
    return DummyVecEnv([lambda: env])


class RLAgent:
    """
    RLAgent encapsulates the policy layer.

    Usage:
        from config import get_default_config
        from env_trading import TradingEnv
        from stats_evidence import EvidenceEngine
        from agent_rl import RLAgent

        cfg = get_default_config()
        engine = EvidenceEngine.from_global_config(cfg)
        env = TradingEnv(features_df, engine, cfg)

        agent = RLAgent(env, cfg)
        agent.train()
        agent.save()

        # Later / in another script:
        agent_loaded = RLAgent.load(env, cfg, path="models/SPY_dqn")
        action = agent_loaded.act(state)
    """

    def __init__(
        self,
        env: TradingEnv,
        cfg: Optional[Config] = None,
    ):
        self.cfg = cfg or get_default_config()
        self.env = env
        self.vec_env = _make_vec_env(env)

        algo_name = self.cfg.rl_algo.lower()
        if algo_name not in ALGO_MAP:
            raise ValueError(
                f"Unknown RL algorithm '{self.cfg.rl_algo}'. "
                f"Supported: {list(ALGO_MAP.keys())}"
            )

        self.algo_name = algo_name
        algo_cls: Type[BaseAlgorithm] = ALGO_MAP[algo_name]

        # Create RL model with basic hyperparameters from config
        # You can expand this later in a structured way.
        if algo_name == "dqn":
            self.model: BaseAlgorithm = algo_cls(
                "MlpPolicy",
                self.vec_env,
                gamma=self.cfg.gamma,
                batch_size=self.cfg.batch_size,
                verbose=self.cfg.verbose,
                seed=self.cfg.seed,
                tensorboard_log=None,
            )
        else:  # "ppo"
            self.model = algo_cls(
                "MlpPolicy",
                self.vec_env,
                gamma=self.cfg.gamma,
                batch_size=self.cfg.batch_size,
                verbose=self.cfg.verbose,
                seed=self.cfg.seed,
                tensorboard_log=None,
            )


    # Training & acting

    def train(self) -> None:
        """
        Train the RL agent on the environment for cfg.total_timesteps.
        """
        self.model.learn(total_timesteps=self.cfg.total_timesteps)

    def act(self, state: np.ndarray, deterministic: bool = True) -> int:
        """
        Get an action from the trained policy for a single state.

        Parameters:
            state: 1D numpy array (observation from env.reset() or env.step()).
            deterministic: if True, use deterministic policy when available.

        Returns:
            action: integer (for Discrete action spaces).
        """
        state = np.array(state, dtype=np.float32)
        # Stable-baselines handles batching internally
        action, _ = self.model.predict(state, deterministic=deterministic)
        # For Discrete spaces, action is a scalar int
        return int(action)

    # Saving & loading

    def _default_model_path(self) -> Path:
        """
        Default path where the model will be saved, based on config.
        """
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Example: models/SPY_dqn
        model_name = f"{self.cfg.symbol}_{self.algo_name}"
        return save_dir / model_name

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.

        Parameters:
            path: optional explicit path (without extension). If None, use
                  a default path based on cfg.save_dir, cfg.symbol, and algo.

        Returns:
            The actual path used (string).
        """
        if path is None:
            model_path = self._default_model_path()
        else:
            model_path = Path(path)

        self.model.save(model_path.as_posix())
        return model_path.as_posix()

    @classmethod
    def load(
        cls,
        env: TradingEnv,
        cfg: Optional[Config] = None,
        path: Optional[str] = None,
    ) -> "RLAgent":
        """
        Load a previously saved model and return an RLAgent instance.

        Parameters:
            env: TradingEnv instance to attach the loaded model to.
            cfg: Config (if None, uses default).
            path: path (without extension) where model was saved. If None,
                  uses the default path as in _default_model_path().

        Returns:
            RLAgent with loaded model.
        """
        cfg = cfg or get_default_config()
        agent = cls(env, cfg)  # this creates vec_env etc.

        if path is None:
            model_path = agent._default_model_path()
        else:
            model_path = Path(path)

        algo_name = cfg.rl_algo.lower()
        if algo_name not in ALGO_MAP:
            raise ValueError(
                f"Unknown RL algorithm '{cfg.rl_algo}' when loading. "
                f"Supported: {list(ALGO_MAP.keys())}"
            )

        algo_cls: Type[BaseAlgorithm] = ALGO_MAP[algo_name]

        # Load model, attach to existing vec_env
        agent.model = algo_cls.load(
            model_path.as_posix(),
            env=agent.vec_env,
        )
        return agent
