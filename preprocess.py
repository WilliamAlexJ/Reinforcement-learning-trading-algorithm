# preprocess.py
"""
Feature engineering for the trading project.

This module does ALL preprocessing:
- Take raw OHLCV data (e.g. from yfinance via data_io.load_raw_prices)
- Clean it
- Compute returns and technical features
- Compute GARCH(1,1) conditional volatility
- Compute basic risk features (e.g. drawdown)
- Provide a train/test split helper

No I/O happens here. Reading/writing is done in data_io.py.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model

from config import Config, get_default_config


#Robust price handling

def _choose_price_column(df: pd.DataFrame) -> pd.Series:
    """
    Choose which price series to treat as the main tradeable price.

    Preference:
    1. 'Adj Close' if available
    2. 'Close' otherwise

    Robust to:
    - yfinance multi-index columns (e.g. ('Adj Close', 'SPY'))
    - stray string entries (e.g. 'SPY') → coerced to NaN and later dropped.

    Always returns a 1D numeric Series named 'price'.
    """
    price = None

    #Simple single-level columns
    if "Adj Close" in df.columns:
        price = df["Adj Close"].copy()
    elif "Close" in df.columns:
        price = df["Close"].copy()

    #If not found, try MultiIndex columns like ('Adj Close', 'SPY')
    if price is None:
        adj_candidates = [
            c for c in df.columns
            if isinstance(c, tuple) and c[0] in ("Adj Close", "Close")
        ]
        if adj_candidates:
            price = df[adj_candidates[0]].copy()

    if price is None:
        raise ValueError(
            "Input DataFrame must contain 'Adj Close' or 'Close' "
            "or a MultiIndex variant of those."
        )

    #If we still have a DataFrame (e.g. multiple tickers), take the first column
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    #Force numeric; anything like 'SPY' becomes NaN and will be dropped later
    price = pd.to_numeric(price, errors="coerce")
    price.name = "price"
    return price


# Core feature functions

def _compute_log_returns(price: pd.Series) -> pd.Series:
    """
    Compute 1-day log returns from a price series.
    """
    ret = np.log(price / price.shift(1))
    ret.name = "ret_1d"
    return ret


def _compute_simple_return(price: pd.Series, lag: int) -> pd.Series:
    """
    Compute simple lagged returns: (P_t / P_{t-lag} - 1)
    """
    r = price / price.shift(lag) - 1.0
    r.name = f"ret_{lag}d"
    return r


def _compute_rolling_volatility(
    ret: pd.Series,
    window: int,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Rolling realized volatility based on returns.

    By default, annualizes using sqrt(252).
    """
    vol = ret.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(trading_days)
    vol.name = f"vol_{window}d"
    return vol


def _compute_moving_average(price: pd.Series, window: int) -> pd.Series:
    """
    Simple moving average of price.
    """
    ma = price.rolling(window=window).mean()
    ma.name = f"ma_{window}d"
    return ma


def _compute_momentum(price: pd.Series, window: int) -> pd.Series:
    """
    Price momentum: P_t / mean(P_{t-window+1:t}) - 1
    """
    avg = price.rolling(window=window).mean()
    mom = price / avg - 1.0
    mom.name = f"mom_{window}d"
    return mom


def _compute_max_drawdown(price: pd.Series) -> pd.Series:
    """
    Compute running max drawdown of a price (or equity) series.

    Drawdown_t = (price_t / rolling_max_t) - 1
    """
    rolling_max = price.cummax()
    dd = price / rolling_max - 1.0
    dd.name = "drawdown"
    return dd


def _compute_garch_vol(ret: pd.Series) -> pd.Series:
    """
    Fit a GARCH(1,1) model on the full return series and return
    conditional volatility (sigma_t).

    We fit once on the whole series (for simplicity) and align back
    to the original index. For very long histories, you might want
    to fit on a rolling window instead.
    """
    r = ret.dropna() * 100.0  # scale to percent for numerical stability

    if len(r) < 50:
        # Not enough data to fit GARCH reliably
        return pd.Series(np.nan, index=ret.index, name="garch_vol")

    am = arch_model(r, vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")

    cond_vol = res.conditional_volatility / 100.0  # back to return units
    cond_vol.name = "garch_vol"
    cond_vol = cond_vol.reindex(ret.index)
    return cond_vol


#Main API

def build_feature_matrix(
    raw_df: pd.DataFrame,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Main entry point: build a feature matrix from raw OHLCV data.

    Input:
        raw_df: DataFrame with datetime index and at least ['Adj Close'] or ['Close'].

    Output:
        features_df: DataFrame indexed by datetime with columns:
            - price          (chosen main price)
            - ret_1d         (log return)
            - ret_5d         (simple 5-day return)
            - ret_21d        (simple 21-day return, ~1 month)
            - vol_21d        (21-day realized vol, annualized)
            - ma_10d         (10-day moving average)
            - ma_50d         (50-day moving average)
            - mom_10d        (10-day momentum)
            - drawdown       (running max drawdown)
            - garch_vol      (GARCH(1,1) conditional volatility)
    """
    if cfg is None:
        cfg = get_default_config()

    if raw_df.index.name is None:
        raw_df = raw_df.copy()
        raw_df.index.name = "Date"

    # Ensure sorted by time
    df = raw_df.sort_index().copy()

    #Choose main price series (robust handling)
    price = _choose_price_column(df)

    #Basic returns
    ret_1d = _compute_log_returns(price)
    ret_5d = _compute_simple_return(price, lag=5)
    ret_21d = _compute_simple_return(price, lag=21)

    #Volatility
    vol_21d = _compute_rolling_volatility(ret_1d, window=21, annualize=True)

    #Moving averages & momentum
    ma_10 = _compute_moving_average(price, window=10)
    ma_50 = _compute_moving_average(price, window=50)
    mom_10 = _compute_momentum(price, window=10)

    # Risk feature: drawdown
    drawdown = _compute_max_drawdown(price)

    #GARCH(1,1) conditional volatility
    garch_vol = _compute_garch_vol(ret_1d)

    # Assemble into one DataFrame
    features_df = pd.concat(
        [
            price,
            ret_1d,
            ret_5d,
            ret_21d,
            vol_21d,
            ma_10,
            ma_50,
            mom_10,
            drawdown,
            garch_vol,
        ],
        axis=1,
    )

    # Drop rows with NaNs created by rolling calculations and coercions
    features_df = features_df.dropna()

    return features_df


def split_train_test(
    features_df: pd.DataFrame,
    cfg: Optional[Config] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenient helper to split the feature matrix into train and test periods.

    If a Config is provided with train_start/train_end/test_start/test_end,
    use those date ranges. Otherwise, fall back to a simple 70/30 split.

    Returns:
        train_df, test_df
    """
    if cfg is None:
        # no config → simple ratio split
        split_idx = int(0.7 * len(features_df))
        train_df = features_df.iloc[:split_idx].copy()
        test_df = features_df.iloc[split_idx:].copy()
        return train_df, test_df

    # With config: use date-based split
    train_df = features_df.loc[cfg.train_start : cfg.train_end].copy()
    test_df = features_df.loc[cfg.test_start : cfg.test_end].copy()

    return train_df, test_df
