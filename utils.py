"""
Shared utilities for the Options Trader pipeline.
Single source of truth for indicator calculations, pattern detection, and data helpers.
"""
import pandas as pd
import numpy as np
import os
import glob


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    """
    Compute RSI using the standard Wilder smoothing method.
    This is the ONLY RSI function — all scripts must use this.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_atr(df, period=14):
    """
    Average True Range — purpose-built volatility measure for OHLC data.
    Far more stable than pct_change().std() for volatility filtering.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr


def calculate_sma(series, period=20):
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Returns (sma, upper_band, lower_band)."""
    sma = calculate_sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return sma, upper, lower


def add_all_indicators(df, sma_period=20, bb_period=20, bb_std=2, rsi_period=14, atr_period=14):
    """
    Add all technical indicators to a DataFrame in-place.
    Standardized column names used across the entire pipeline.
    """
    # Bollinger Bands (also gives us SMA)
    df['SMA_20'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(
        df['Close'], period=bb_period, std_dev=bb_std
    )

    # RSI
    df['RSI_14'] = calculate_rsi(df['Close'], period=rsi_period)

    # ATR
    df['ATR_14'] = calculate_atr(df, period=atr_period)

    # Volume ratio (current volume / 20-period avg volume)
    vol_sma = df['Volume'].rolling(window=20, min_periods=20).mean()
    df['Volume_Ratio'] = df['Volume'] / vol_sma

    # Returns (1-period percentage change)
    df['Returns'] = df['Close'].pct_change()

    # Candlestick anatomy features
    body = (df['Close'] - df['Open']).abs()
    full_range = df['High'] - df['Low']
    df['Body_Ratio'] = body / full_range.replace(0, np.nan)
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / full_range.replace(0, np.nan)
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / full_range.replace(0, np.nan)

    return df


# ============================================================
# PATTERN DETECTION
# ============================================================

def detect_bullish_engulfing(df):
    """
    Vectorized bullish engulfing detection.
    Returns a boolean Series.
    """
    direction = np.where(df['Close'] >= df['Open'], 1, -1)
    prev_dir = np.roll(direction, 1)
    prev_dir[0] = 0  # invalid first element

    prev_red = prev_dir == -1
    curr_green = direction == 1
    engulfs_top = df['Close'].values > df['Open'].shift(1).values
    engulfs_bottom = df['Open'].values < df['Close'].shift(1).values

    pattern = prev_red & curr_green & engulfs_top & engulfs_bottom
    return pd.Series(pattern, index=df.index)


def detect_hammer(df, body_ratio_max=0.3, lower_shadow_min=0.6):
    """
    Hammer pattern: small body at top, long lower shadow.
    Bullish reversal signal.
    """
    body = (df['Close'] - df['Open']).abs()
    full_range = (df['High'] - df['Low']).replace(0, np.nan)
    body_ratio = body / full_range
    lower_shadow = (df[['Open', 'Close']].min(axis=1) - df['Low']) / full_range

    return (body_ratio < body_ratio_max) & (lower_shadow > lower_shadow_min)


def detect_doji(df, body_ratio_max=0.05):
    """
    Doji: open ≈ close (tiny body), signaling indecision.
    """
    body = (df['Close'] - df['Open']).abs()
    full_range = (df['High'] - df['Low']).replace(0, np.nan)
    body_ratio = body / full_range
    return body_ratio < body_ratio_max


def detect_patterns_with_confirmation(df, volume_multiplier=1.5):
    """
    Detect patterns with volume confirmation.
    Only flags patterns where volume exceeds the 20-period average by the multiplier.
    Returns a boolean Series.
    """
    engulfing = detect_bullish_engulfing(df)
    hammer = detect_hammer(df)

    # Volume confirmation
    vol_sma = df['Volume'].rolling(window=20, min_periods=20).mean()
    vol_confirmed = df['Volume'] > (vol_sma * volume_multiplier)

    # Trend context: pattern near lower Bollinger Band is stronger
    near_lower_bb = df['Close'] <= df.get('BB_Lower', df['Close']) * 1.01

    # Combined: pattern detected AND (volume confirmed OR near support)
    return (engulfing | hammer) & (vol_confirmed | near_lower_bb)


# ============================================================
# RISK-ADJUSTED LABELING
# ============================================================

def create_risk_adjusted_labels(df, look_ahead=15, profit_target=0.003, rr_ratio=1.5):
    """
    Labels that account for both upside AND downside potential.

    A sample is labeled '1' (Buy) only if:
      - Max gain in the next `look_ahead` candles exceeds `profit_target`
      - Max gain exceeds max loss by at least `rr_ratio`

    This eliminates the bullish bias of the old "max high" approach.
    """
    # Forward-looking max gain and max loss
    future_high = df['High'].rolling(look_ahead).max().shift(-look_ahead)
    future_low = df['Low'].rolling(look_ahead).min().shift(-look_ahead)

    max_gain = (future_high - df['Close']) / df['Close']
    max_loss = (df['Close'] - future_low) / df['Close']

    # Label '1' only if:  gain > target  AND  gain > loss * rr_ratio
    label = np.where(
        (max_gain > profit_target) & (max_gain > max_loss * rr_ratio),
        1, 0
    )

    return pd.Series(label, index=df.index), max_gain, max_loss


# ============================================================
# DATA SPLIT UTILITIES
# ============================================================

def get_chronological_split(directory, split_ratio=0.8, file_pattern="*.png"):
    """
    Split image dataset chronologically by extracting numeric IDs from filenames.
    Returns (train_df, val_df, class_weight_dict).
    """
    from sklearn.utils import class_weight as cw

    filepaths = []
    labels = []
    for label in ['0', '1']:
        class_dir = os.path.join(directory, label)
        if not os.path.exists(class_dir):
            continue
        files = glob.glob(os.path.join(class_dir, file_pattern))
        for f in files:
            filepaths.append(f)
            labels.append(label)

    df = pd.DataFrame({'filename': filepaths, 'class': labels})

    # Extract numeric ID for chronological sorting
    try:
        df['id'] = df['filename'].apply(
            lambda x: int(os.path.basename(x).split('_')[-1].replace('.png', ''))
        )
    except (ValueError, IndexError) as e:
        print(f"Filename parsing error: {e}")
        return None, None, None

    df = df.sort_values('id').reset_index(drop=True)

    # Chronological split
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    print(f"Training Samples: {len(train_df)} | Validation Samples: {len(val_df)}")

    if len(train_df) == 0:
        print("ERROR: No training samples found. Generate the image dataset first (step2_pro_gen.py).")
        return None, None, None

    # Class weights
    y_train_int = train_df['class'].astype(int).values
    unique_classes = np.unique(y_train_int)

    if len(unique_classes) < 2:
        print(f"WARNING: Only one class ({unique_classes}) found in training data. Using equal weights.")
        weight_dict = {0: 1.0, 1: 1.0}
    else:
        class_weights = cw.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train_int
        )
        weight_dict = dict(enumerate(class_weights))
        print(f"Class Weights — 0 (Wait): {weight_dict.get(0, 'N/A'):.4f}, "
              f"1 (Buy): {weight_dict.get(1, 'N/A'):.4f}")

    return train_df, val_df, weight_dict
