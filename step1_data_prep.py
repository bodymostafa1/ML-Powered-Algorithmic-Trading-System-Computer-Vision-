"""
Step 1: Fetch historical BTC/USDT data from Binance and apply risk-adjusted labeling.

Changes from original:
  - Uses shared config & utils
  - Risk-adjusted labels (reward/risk ratio, not just direction)
  - Adds all indicators during data prep so downstream scripts don't recalculate
  - Reproducible with seeded randomness
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time

from config import (
    SYMBOL, TIMEFRAME, SINCE_DATE, INPUT_FILE, SEED,
    LOOK_AHEAD, LABEL_THRESHOLD, PROFIT_TARGET, REWARD_RISK_RATIO,
    set_seeds
)
from utils import add_all_indicators, create_risk_adjusted_labels

set_seeds(SEED)


def fetch_historical_data(symbol, timeframe, start_date_str):
    """Fetch OHLCV data from Binance in paginated batches."""
    print(f"--- Connecting to Binance to fetch {symbol} since {start_date_str} ---")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date_str)
    all_candles = []

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles:
                break

            all_candles.extend(candles)

            last_time = candles[-1][0]
            since = last_time + 60000  # Next minute

            last_date = datetime.fromtimestamp(last_time / 1000)
            print(f"Fetched up to: {last_date} | Total candles: {len(all_candles)}")

            if last_time >= exchange.milliseconds() - 60000:
                break

            time.sleep(0.1)  # Rate limit courtesy

        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df


def label_data(df):
    """
    Apply risk-adjusted labeling strategy.

    Two labeling columns are created:
      - 'Target_Dir': Directional label (up/down beyond threshold) — for general models
      - 'Target_RR':  Risk-adjusted label (reward > risk * ratio) — for pattern models
    """
    print("\n--- Applying Risk-Adjusted Labeling ---")

    # --- Directional labels (with dead zone filtering) ---
    df['Future_Close'] = df['Close'].shift(-LOOK_AHEAD)
    df['Price_Change'] = (df['Future_Close'] - df['Close']) / df['Close']

    conditions_dir = [
        (df['Price_Change'] > LABEL_THRESHOLD),
        (df['Price_Change'] < -LABEL_THRESHOLD)
    ]
    df['Target_Dir'] = np.select(conditions_dir, [1, 0], default=2)

    # --- Risk-adjusted labels ---
    df['Target_RR'], df['Max_Gain'], df['Max_Loss'] = create_risk_adjusted_labels(
        df, look_ahead=LOOK_AHEAD, profit_target=PROFIT_TARGET, rr_ratio=REWARD_RISK_RATIO
    )

    # Primary target: use risk-adjusted label, filter out flat markets from directional
    df['Target'] = df['Target_RR']

    # Drop rows where the directional label was flat (no meaningful move)
    clean_df = df[df['Target_Dir'] != 2].dropna()

    return clean_df


if __name__ == "__main__":
    # 1. Fetch
    df = fetch_historical_data(SYMBOL, TIMEFRAME, SINCE_DATE)

    # 2. Add indicators (once, saved to CSV for all downstream scripts)
    print("\n--- Computing Technical Indicators ---")
    df = add_all_indicators(df)

    # 3. Label
    labeled_df = label_data(df)

    # 4. Report
    print(f"\n--- Final Dataset Report ---")
    print(f"Raw Candles Downloaded: {len(df)}")
    print(f"Training Samples (after filtering): {len(labeled_df)}")

    if len(labeled_df) > 0:
        print(f"\nRisk-Adjusted Label Distribution:")
        print(labeled_df['Target'].value_counts(normalize=True))
        print(f"\nDirectional Label Distribution:")
        print(labeled_df['Target_Dir'].value_counts(normalize=True))

        # Save
        labeled_df.to_csv(INPUT_FILE)
        print(f"\n[Success] Data saved to {INPUT_FILE}. Ready for Step 2!")
    else:
        print("No valid samples found. Check your thresholds.")