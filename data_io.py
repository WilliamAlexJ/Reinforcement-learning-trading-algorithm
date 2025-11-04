# data_io.py
"""
Simple data I/O utilities for the trading project.

This module handles:
- Downloading raw price data from yfinance.
- Caching raw data to CSV.
- Saving/loading processed feature data.

All cleaning / feature engineering happens in `preprocess.py`.

Requires:
    pip install yfinance pandas
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import get_default_config, Config


def _ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_raw_prices(
    cfg: Optional[Config] = None,
    symbol: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load raw price data for a given symbol using yfinance.

    If use_cache=True, this will:
      - First look for data/raw/{symbol}.csv
      - If found, load from CSV.
      - If not found, download from yfinance and save to CSV.

    If use_cache=False, it always downloads fresh data for the requested
    date range (train_start to test_end in Config) and does NOT save.

    Expected columns from yfinance:
      'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'

    Returns:
    - DataFrame indexed by datetime (sorted ascending), with OHLCV columns.

    Parameters:
    - cfg: optional Config object (uses default if None).
    - symbol: optional override of cfg.symbol.
    - use_cache: whether to use / create CSV cache in data/raw.
    """
    if cfg is None:
        cfg = get_default_config()

    if symbol is None:
        symbol = cfg.symbol

    raw_dir = Path(cfg.data_path_raw)
    _ensure_dir(raw_dir)

    csv_path = raw_dir / f"{symbol}.csv"

    #Try to use cached file
    if use_cache and csv_path.exists():
        print(f"[data_io] Loading cached file: {csv_path}")
        try:
            # Preferred format: CSV with a 'Date' column
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.sort_values("Date").set_index("Date")
            return df
        except ValueError:
            # Old format: index saved without explicit 'Date' column
            # Fallback: treat first column as the datetime index
            print("[data_io] Cached file missing 'Date' column, "
                  "attempting fallback load and rewriting cache...")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = df.sort_index()
            df.index.name = "Date"

            # Rewrite cache in new, clean format with 'Date' column
            df_reset = df.reset_index()
            df_reset.to_csv(csv_path, index=False)
            print("[data_io] Rewrote cached file with explicit 'Date' column.")

            return df

    #Download fresh from yfinance
    print(f"[data_io] Downloading {symbol} from yfinance...")
    start_date = cfg.train_start
    end_date = cfg.test_end

    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data returned from yfinance for symbol {symbol}.")

    # Ensure sorted and consistent index naming
    df = df.sort_index()
    df.index.name = "Date"

    # Save to cache in a robust format: 'Date' as a column
    if use_cache:
        df_reset = df.reset_index()
        df_reset.to_csv(csv_path, index=False)
        print(f"[data_io] Saved raw prices to {csv_path}")

    return df


def save_processed_features(
    features_df: pd.DataFrame,
    cfg: Optional[Config] = None,
    symbol: Optional[str] = None,
) -> None:
    """
    Save processed feature matrix to disk as CSV.

    Output path:
        data/processed/{symbol}_features.csv

    Parameters:
    - features_df: DataFrame with datetime index and feature columns.
    - cfg: optional Config object (uses default if None).
    - symbol: optional override of cfg.symbol.
    """
    if cfg is None:
        cfg = get_default_config()

    if symbol is None:
        symbol = cfg.symbol

    processed_dir = Path(cfg.data_path_processed)
    _ensure_dir(processed_dir)

    file_path = processed_dir / f"{symbol}_features.csv"

    df_to_save = features_df.copy()
    if df_to_save.index.name is None:
        df_to_save.index.name = "Date"

    # Save with 'Date' as a column
    df_to_save.reset_index().to_csv(file_path, index=False)
    print(f"[data_io] Saved processed features to {file_path}")


def load_processed_features(
    cfg: Optional[Config] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load processed feature matrix from disk.

    Input path:
        data/processed/{symbol}_features.csv

    Returns:
    - DataFrame indexed by datetime.

    Parameters:
    - cfg: optional Config object (uses default if None).
    - symbol: optional override of cfg.symbol.
    """
    if cfg is None:
        cfg = get_default_config()

    if symbol is None:
        symbol = cfg.symbol

    processed_dir = Path(cfg.data_path_processed)
    file_path = processed_dir / f"{symbol}_features.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed features file not found: {file_path}\n"
            f"Run your preprocessing step first to create it."
        )

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    return df
