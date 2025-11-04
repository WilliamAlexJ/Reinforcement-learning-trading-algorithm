# stats_evidence.py
"""
Statistical Evidence Layer with ML-based likelihood tests, GARCH awareness,
and regime classification.

Outputs:
- evidence_strength (model improvement)
- signal_valid
- dist_shift_flag (KS test)
- regime_type (0=normal, 1=high_vol, 2=shifted)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import Config, get_default_config


@dataclass
class EvidenceConfig:
    rolling_window: int
    min_samples_for_tests: int
    evidence_score_threshold: float
    ks_pvalue_threshold: float
    signal_col: str = "mom_10d"


class EvidenceEngine:
    def __init__(self, ev_cfg: EvidenceConfig):
        self.ev_cfg = ev_cfg

    @classmethod
    def from_global_config(
        cls,
        cfg: Optional[Config] = None,
        signal_col: str = "mom_10d",
    ) -> "EvidenceEngine":
        if cfg is None:
            cfg = get_default_config()
        ev_cfg = EvidenceConfig(
            rolling_window=cfg.rolling_window,
            min_samples_for_tests=cfg.min_samples_for_tests,
            evidence_score_threshold=cfg.lr_pvalue_threshold,  # reusing param slot
            ks_pvalue_threshold=cfg.ks_pvalue_threshold,
            signal_col=signal_col,
        )
        return cls(ev_cfg)

    # Public API

    def compute_tests(self, history_window: pd.DataFrame) -> Dict[str, float]:
        n = len(history_window)
        if n < self.ev_cfg.min_samples_for_tests:
            return self._neutral_evidence()

        # Required columns
        if "ret_1d" not in history_window.columns:
            raise ValueError("Missing 'ret_1d'")
        if self.ev_cfg.signal_col not in history_window.columns:
            raise ValueError(f"Missing signal column {self.ev_cfg.signal_col}")

        ret = history_window["ret_1d"].astype(float)
        signal = history_window[self.ev_cfg.signal_col].astype(float)
        vol = history_window.get("garch_vol", pd.Series(np.zeros(n), index=ret.index))

        #ML-based “Likelihood Ratio” Test 
        imp, mse_null, mse_alt = self._ml_signal_test(signal, ret)
        signal_valid = 1.0 if imp > self.ev_cfg.evidence_score_threshold else 0.0
        evidence_strength = max(imp, 0.0)

        # KS test for distribution shift
        ks_stat, ks_pvalue = self._ks_shift_test(ret)
        dist_shift_flag = 1.0 if ks_pvalue < self.ev_cfg.ks_pvalue_threshold else 0.0

        #Volatility regime classification
        vol_ratio = float(vol.iloc[-1] / (vol.mean() + 1e-8))
        high_vol_flag = 1.0 if vol_ratio > 1.5 else 0.0

        # Regime logic
        if dist_shift_flag and high_vol_flag:
            regime = 2.0  # shifted + high vol
        elif high_vol_flag:
            regime = 1.0  # high vol
        else:
            regime = 0.0  # normal

        return {
            "signal_valid": signal_valid,
            "evidence_strength": evidence_strength,
            "dist_shift_flag": dist_shift_flag,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "garch_vol_ratio": vol_ratio,
            "high_vol_flag": high_vol_flag,
            "regime_type": regime,
            "ml_mse_null": mse_null,
            "ml_mse_alt": mse_alt,
        }

    #Internal 

    @staticmethod
    def _neutral_evidence():
        return {
            "signal_valid": 0.0,
            "evidence_strength": 0.0,
            "dist_shift_flag": 0.0,
            "ks_stat": 0.0,
            "ks_pvalue": 1.0,
            "garch_vol_ratio": 1.0,
            "high_vol_flag": 0.0,
            "regime_type": 0.0,
            "ml_mse_null": np.nan,
            "ml_mse_alt": np.nan,
        }

    def _ml_signal_test(self, signal: pd.Series, ret: pd.Series):
        """Train ML regressors to check if signal adds predictive power."""
        X_signal = signal.values.reshape(-1, 1)
        X_null = np.ones((len(signal), 1))
        y = ret.values

        # ML models
        rf_null = RandomForestRegressor(n_estimators=50, random_state=0)
        rf_alt = RandomForestRegressor(n_estimators=50, random_state=0)

        # Fit both
        rf_null.fit(X_null, y)
        rf_alt.fit(X_signal, y)

        # Predict
        y_null = rf_null.predict(X_null)
        y_alt = rf_alt.predict(X_signal)

        # Compute MSE
        mse_null = mean_squared_error(y, y_null)
        mse_alt = mean_squared_error(y, y_alt)
        improvement = mse_null - mse_alt  # larger = signal improves fit
        return improvement, mse_null, mse_alt

    def _ks_shift_test(self, ret: pd.Series):
        """KS test between first 2/3 and last 1/3 of window."""
        r = ret.dropna().values
        n = len(r)
        if n < self.ev_cfg.min_samples_for_tests:
            return 0.0, 1.0
        split = int(2 * n / 3)
        ks = ks_2samp(r[:split], r[split:], mode="auto")
        return ks.statistic, ks.pvalue
