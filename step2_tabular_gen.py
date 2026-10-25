"""
Step 2B: Generate tabular (numeric sequence) data for LSTM/1D-CNN training.

This is the HIGH-IMPACT addition from the algorithmic review:
  - 60-candle sequences of exact floating-point features
  - No image rendering bottleneck (100x faster than image pipeline)
  - Model receives precise indicator values instead of pixel approximations

Output: .npz files containing X (sequences) and y (labels) arrays.
"""
import pandas as pd
import numpy as np
import os

from config import (
    INPUT_FILE, TABULAR_DATASET_DIR, SEQUENCE_LENGTH, TABULAR_FEATURES,
    TRAIN_SPLIT, SEED, set_seeds
)
from utils import add_all_indicators, detect_patterns_with_confirmation

set_seeds(SEED)


def normalize_window(window_df, features):
    """
    Normalize a single window using min-max scaling per feature.
    Each feature is scaled to [0, 1] within the window context.
    This makes the model invariant to absolute price levels.
    """
    arr = window_df[features].values.astype(np.float32)
    mins = arr.min(axis=0, keepdims=True)
    maxs = arr.max(axis=0, keepdims=True)
    ranges = maxs - mins
    # Avoid division by zero (constant features get 0.5)
    ranges[ranges == 0] = 1.0
    normalized = (arr - mins) / ranges
    return normalized


def generate_tabular_dataset():
    """Generate sequences of normalized features at pattern locations."""
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Ensure indicators exist
    if 'SMA_20' not in df.columns:
        print("Indicators not found, recalculating...")
        df = add_all_indicators(df)
        print("Saving indicators back to CSV for future runs...")
        df.to_csv(INPUT_FILE)
        print("Saved.")

    # Detect patterns
    df['Pattern'] = detect_patterns_with_confirmation(df)

    if 'Target' not in df.columns:
        print("ERROR: 'Target' column not found. Run step1_data_prep.py first.")
        return

    df.dropna(subset=TABULAR_FEATURES + ['Target'], inplace=True)

    # Only generate sequences at pattern locations
    pattern_indices = df[df['Pattern']].index
    print(f"Found {len(pattern_indices)} pattern setups for tabular data.")

    X_all = []
    y_all = []
    ids = []  # For chronological ordering

    for idx, timestamp in enumerate(pattern_indices):
        try:
            i = df.index.get_loc(timestamp)
            if i < SEQUENCE_LENGTH:
                continue

            window = df.iloc[i - SEQUENCE_LENGTH + 1: i + 1]

            # Normalize per-window
            seq = normalize_window(window, TABULAR_FEATURES)
            target = int(df.loc[timestamp]['Target'])

            X_all.append(seq)
            y_all.append(target)
            ids.append(i)

        except Exception as e:
            continue

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int32)
    ids = np.array(ids, dtype=np.int64)

    print(f"Generated {len(X_all)} sequences of shape {X_all.shape[1:]}")
    print(f"  Features: {TABULAR_FEATURES}")
    print(f"  Class distribution — 0: {(y_all == 0).sum()} | 1: {(y_all == 1).sum()}")

    # Chronological split
    sort_order = np.argsort(ids)
    X_all = X_all[sort_order]
    y_all = y_all[sort_order]

    split_idx = int(len(X_all) * TRAIN_SPLIT)
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    print(f"\nChronological Split:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # Save
    os.makedirs(TABULAR_DATASET_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(TABULAR_DATASET_DIR, "train.npz"),
        X=X_train, y=y_train
    )
    np.savez_compressed(
        os.path.join(TABULAR_DATASET_DIR, "val.npz"),
        X=X_val, y=y_val
    )

    print(f"\n[Done] Saved to '{TABULAR_DATASET_DIR}/train.npz' and 'val.npz'")


if __name__ == "__main__":
    generate_tabular_dataset()
