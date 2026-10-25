"""
Step 4: Live trading bot with ensemble predictions (CNN + Tabular).

Changes from original:
  - ATR-based volatility filter instead of naive std() calculation
  - Ensemble: combines CNN (image) + Tabular (numeric) predictions
  - Volume-confirmed pattern detection with multiple pattern types
  - Shared config and utils
  - Proper logging with structured output
  - Tabular model is primary (weighted 60%), CNN is secondary (40%)
"""
import ccxt
import pandas as pd
import mplfinance as mpf
import tensorflow as tf
import numpy as np
import os
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from config import (
    SYMBOL, IMG_WINDOW, IMG_DIM, CNN_MODEL_PATH, TABULAR_MODEL_PATH,
    CONFIDENCE_THRESHOLD, VOLUME_CONFIRMATION_MULTIPLIER,
    ATR_VOLATILITY_PERIODS, MIN_ATR_PERCENTILE,
    SEQUENCE_LENGTH, TABULAR_FEATURES, ENSEMBLE_WEIGHTS, SEED, set_seeds
)
from utils import (
    add_all_indicators, detect_patterns_with_confirmation, calculate_atr
)
from step3_train_tabular import SelfAttention

set_seeds(SEED)

# --- LOGGING ---
logging.basicConfig(
    filename='trade_log.txt',
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)


def load_models():
    """Load both models, gracefully handle missing ones."""
    models = {}

    if os.path.exists(CNN_MODEL_PATH):
        models['cnn'] = tf.keras.models.load_model(CNN_MODEL_PATH)
        print(f"  ✅ CNN model loaded: {CNN_MODEL_PATH}")
    else:
        print(f"  ⚠️ CNN model not found: {CNN_MODEL_PATH}")

    if os.path.exists(TABULAR_MODEL_PATH):
        models['tabular'] = tf.keras.models.load_model(
            TABULAR_MODEL_PATH, custom_objects={'SelfAttention': SelfAttention}
        )
        print(f"  ✅ Tabular model loaded: {TABULAR_MODEL_PATH}")
    else:
        print(f"  ⚠️ Tabular model not found: {TABULAR_MODEL_PATH}")

    if not models:
        print("  ❌ No models found! Train at least one model first.")
        return None

    return models


def preprocess_live_data(exchange):
    """Fetch live candles and compute all indicators."""
    candles = exchange.fetch_ohlcv(SYMBOL, timeframe='1m', limit=120)
    df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    # Add all indicators using shared utils
    df = add_all_indicators(df)
    return df


def check_volatility(df):
    """
    ATR-based volatility filter.
    Returns (is_volatile, current_atr, atr_percentile).
    """
    atr = df['ATR_14'].dropna()
    if len(atr) < 20:
        return True, 0, 50  # Not enough data, allow trading

    current_atr = atr.iloc[-1]
    percentile = (atr < current_atr).sum() / len(atr) * 100

    return percentile > MIN_ATR_PERCENTILE, current_atr, percentile


def generate_live_image(df):
    """Render a chart image for the CNN model."""
    temp_img = "live_chart.png"

    if os.path.exists(temp_img):
        try:
            os.remove(temp_img)
        except PermissionError:
            return temp_img

    plt.close('all')

    window = df.iloc[-IMG_WINDOW:]
    mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', edge='inherit', wick='inherit', volume='in')
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle='',
        rc={'axes.axisbelow': True, 'figure.facecolor': 'black',
            'axes.facecolor': 'black', 'axes.edgecolor': 'black'}
    )

    apds = [
        mpf.make_addplot(window['SMA_20'], color='blue', width=1.5, panel=0),
        mpf.make_addplot(window['BB_Upper'], color='white', width=0.5, linestyle='dotted', panel=0),
        mpf.make_addplot(window['BB_Lower'], color='white', width=0.5, linestyle='dotted', panel=0),
    ]
    if 'RSI_14' in window.columns and window['RSI_14'].notna().any():
        apds.append(
            mpf.make_addplot(window['RSI_14'], color='purple', width=1.0, panel=2, ylim=(0, 100))
        )

    plt.switch_backend('Agg')
    try:
        mpf.plot(window, type='candle', style=style, addplot=apds,
                 volume=True, axisoff=True,
                 savefig=dict(fname=temp_img, dpi=75, bbox_inches='tight', pad_inches=0),
                 returnfig=True, closefig=True,
                 panel_ratios=(3, 1, 1))
    except Exception:
        return None

    plt.close('all')
    return temp_img


def predict_cnn(model, df):
    """Generate CNN prediction from chart image."""
    img_file = generate_live_image(df)
    if img_file is None:
        return None

    img = tf.keras.preprocessing.image.load_img(img_file, target_size=IMG_DIM)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0) / 255.0

    return model.predict(img_arr, verbose=0)[0][0]


def predict_tabular(model, df):
    """Generate tabular prediction from numeric sequence."""
    window = df.iloc[-SEQUENCE_LENGTH:]

    # Check we have all features
    available = [f for f in TABULAR_FEATURES if f in window.columns]
    if len(available) < len(TABULAR_FEATURES):
        return None

    arr = window[TABULAR_FEATURES].values.astype(np.float32)

    # Normalize (same as training)
    mins = arr.min(axis=0, keepdims=True)
    maxs = arr.max(axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized = (arr - mins) / ranges

    seq = np.expand_dims(normalized, axis=0)
    return model.predict(seq, verbose=0)[0][0]


def ensemble_predict(models, df):
    """
    Weighted ensemble prediction.
    Returns (final_probability, individual_predictions_dict).
    """
    preds = {}
    weights = {}

    if 'cnn' in models:
        p = predict_cnn(models['cnn'], df)
        if p is not None:
            preds['cnn'] = p
            weights['cnn'] = ENSEMBLE_WEIGHTS['cnn']

    if 'tabular' in models:
        p = predict_tabular(models['tabular'], df)
        if p is not None:
            preds['tabular'] = p
            weights['tabular'] = ENSEMBLE_WEIGHTS['tabular']

    if not preds:
        return None, {}

    # Weighted average
    total_weight = sum(weights.values())
    final = sum(preds[k] * weights[k] for k in preds) / total_weight

    return final, preds


def run_bot():
    """Main trading loop."""
    print("=" * 60)
    print("  🚀 ENSEMBLE AI TRADER (CNN + Tabular)")
    print("=" * 60)
    print(f"  Symbol: {SYMBOL}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Ensemble Weights: CNN={ENSEMBLE_WEIGHTS['cnn']}, "
          f"Tabular={ENSEMBLE_WEIGHTS['tabular']}")
    print()

    print("Loading models...")
    models = load_models()
    if models is None:
        return

    exchange = ccxt.binance()

    # Verify connection
    print("\nTesting Binance connection...", end=" ")
    try:
        exchange.fetch_ticker(SYMBOL)
        print("Success! ✅")
    except Exception as e:
        print(f"Failed! ❌ Error: {e}")
        return

    print("\nBot active. Scanning for patterns...\n")

    while True:
        try:
            # Sync with minute boundary
            now = datetime.now()
            sleep_time = 60 - now.second
            if sleep_time > 0:
                print(f"[{now.strftime('%H:%M:%S')}] Next scan in {sleep_time}s...")
                time.sleep(sleep_time)

            # 1. Get Data
            df = preprocess_live_data(exchange)
            current_price = df['Close'].iloc[-1]
            timestamp = datetime.now().strftime("%H:%M:%S")

            # 2. ATR Volatility Filter
            is_volatile, atr_val, atr_pct = check_volatility(df)
            if not is_volatile:
                print(f"[{timestamp}] 💤 Low volatility (ATR={atr_val:.2f}, "
                      f"pctl={atr_pct:.0f}%). Sleeping...")
                continue

            # 3. Pattern Detection
            df['Pattern'] = detect_patterns_with_confirmation(
                df, volume_multiplier=VOLUME_CONFIRMATION_MULTIPLIER
            )

            if df['Pattern'].iloc[-1]:
                print(f"[{timestamp}] 🚨 PATTERN DETECTED @ ${current_price:.2f}")

                # 4. Ensemble Prediction
                probability, individual = ensemble_predict(models, df)

                if probability is None:
                    print(f"  ⚠️ Prediction failed (data issue)")
                    continue

                detail = " | ".join(f"{k}={v:.4f}" for k, v in individual.items())
                print(f"  📊 Predictions: {detail} → Ensemble: {probability:.4f}")

                # 5. Trade Decision
                if probability > CONFIDENCE_THRESHOLD:
                    msg = f"🟢 CALL (UP) | Price: ${current_price:.2f} | Conf: {probability:.4f}"
                    print(f"  {msg}")
                    logging.info(msg)
                elif probability < (1 - CONFIDENCE_THRESHOLD):
                    msg = f"🔴 PUT (DOWN) | Price: ${current_price:.2f} | Conf: {probability:.4f}"
                    print(f"  {msg}")
                    logging.info(msg)
                else:
                    print(f"  ⚪ HOLD — Confidence too low ({probability:.4f})")
            else:
                print(f"[{timestamp}] Scanning... ${current_price:.2f} "
                      f"(ATR={atr_val:.2f}, pctl={atr_pct:.0f}%)")

        except Exception as e:
            print(f"Error: {e}")
            logging.error(str(e))
            time.sleep(10)


if __name__ == "__main__":
    run_bot()