"""
Step 6: Threshold optimization with profit/loss simulation.

Changes from original:
  - Tests both CNN and Tabular models independently
  - Adds P/L simulation at each threshold (not just precision)
  - Tracks consecutive losses for risk assessment
  - Evaluates at multiple payout ratios
  - Reports optimal threshold per model
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os

from config import (
    IMAGE_DATASET_DIR, TABULAR_DATASET_DIR, IMG_DIM, BATCH_SIZE,
    CNN_MODEL_PATH, TABULAR_MODEL_PATH, TRAIN_SPLIT, SEED, set_seeds
)
from utils import get_chronological_split
from step3_train_tabular import SelfAttention

set_seeds(SEED)


def simulate_trading(y_true, y_pred_binary, payout=0.80):
    """
    Simulate binary options trading and return key metrics.
    Win = +payout, Loss = -1.0
    """
    trades = y_pred_binary.sum()
    if trades == 0:
        return {'trades': 0, 'wins': 0, 'win_rate': 0,
                'profit': 0, 'ev_per_trade': 0, 'max_consecutive_losses': 0}

    wins = ((y_pred_binary == 1) & (y_true == 1)).sum()
    losses = trades - wins
    win_rate = wins / trades
    profit = (wins * payout) - (losses * 1.0)
    ev = profit / trades

    # Max consecutive losses
    trade_mask = y_pred_binary == 1
    outcomes = (y_true[trade_mask] == 1).astype(int)
    max_consec_loss = 0
    current_streak = 0
    for outcome in outcomes:
        if outcome == 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    return {
        'trades': int(trades),
        'wins': int(wins),
        'win_rate': win_rate,
        'profit': profit,
        'ev_per_trade': ev,
        'max_consecutive_losses': max_consec_loss
    }


def optimize_for_model(y_true, raw_preds, model_name, payouts=(0.70, 0.80, 0.85)):
    """Test every threshold and find optimal for each payout ratio."""
    print(f"\n{'=' * 80}")
    print(f"  THRESHOLD OPTIMIZATION — {model_name}")
    print(f"{'=' * 80}")

    thresholds = np.arange(0.50, 0.96, 0.05)

    for payout in payouts:
        print(f"\n--- Payout: {payout:.0%} ---")
        header = (f"{'Thresh':<8} | {'Win Rate':<10} | {'Trades':<8} | "
                  f"{'Profit':<10} | {'EV/Trade':<10} | {'Max Loss Streak':<15}")
        print(header)
        print("-" * 75)

        best_profit = -999
        best_thresh = 0.50

        for thresh in thresholds:
            y_pred = (raw_preds >= thresh).astype(int)
            stats = simulate_trading(y_true, y_pred, payout)

            print(f"{thresh:<8.2f} | {stats['win_rate']:<10.2%} | "
                  f"{stats['trades']:<8d} | {stats['profit']:<+10.2f} | "
                  f"{stats['ev_per_trade']:<+10.4f} | {stats['max_consecutive_losses']:<15d}")

            # Best = highest profit with at least 10 trades
            if stats['trades'] >= 10 and stats['profit'] > best_profit:
                best_profit = stats['profit']
                best_thresh = thresh

        print("-" * 75)
        print(f"  BEST -> Threshold: {best_thresh:.2f} | Profit: {best_profit:+.2f}")

    return best_thresh


def optimize_cnn():
    """Optimize threshold for CNN model."""
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"CNN model not found: {CNN_MODEL_PATH}")
        return

    train_df, val_df, _ = get_chronological_split(IMAGE_DATASET_DIR, split_ratio=TRAIN_SPLIT)
    if val_df is None:
        return

    datagen = ImageDataGenerator(rescale=1. / 255)
    val_gen = datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=IMG_DIM, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    preds = model.predict(val_gen, verbose=1).ravel()
    y_true = val_gen.classes

    return optimize_for_model(y_true, preds, "CNN (EfficientNetB0)")


def optimize_tabular():
    """Optimize threshold for tabular model."""
    if not os.path.exists(TABULAR_MODEL_PATH):
        print(f"Tabular model not found: {TABULAR_MODEL_PATH}")
        return

    val_path = os.path.join(TABULAR_DATASET_DIR, "val.npz")
    if not os.path.exists(val_path):
        return

    val_data = np.load(val_path)
    X_val, y_val = val_data['X'], val_data['y']

    model = tf.keras.models.load_model(TABULAR_MODEL_PATH, custom_objects={'SelfAttention': SelfAttention})
    preds = model.predict(X_val, verbose=1).ravel()

    return optimize_for_model(y_val, preds, "Tabular (CNN+BiLSTM)")


if __name__ == "__main__":
    optimize_cnn()
    optimize_tabular()
    print("\n[Done] Use the recommended thresholds in config.py -> CONFIDENCE_THRESHOLD")