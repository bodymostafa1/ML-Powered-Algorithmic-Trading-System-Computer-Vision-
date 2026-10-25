"""
Step 7: Walk-forward backtester.

Tests model performance across rolling time windows to detect:
  - Regime changes (trending vs ranging markets)
  - Performance degradation over time
  - Time-of-day effects
  - Realistic out-of-sample performance

"""
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.metrics import precision_score, recall_score, accuracy_score

from config import (
    INPUT_FILE, TABULAR_FEATURES, SEQUENCE_LENGTH,
    CONFIDENCE_THRESHOLD, SEED, set_seeds
)
from utils import (
    add_all_indicators, detect_patterns_with_confirmation,
    create_risk_adjusted_labels
)

set_seeds(SEED)


def prepare_sequences(df, indices, features, seq_length):
    """Convert pattern indices into normalized sequences."""
    X, y = [], []
    for timestamp in indices:
        try:
            i = df.index.get_loc(timestamp)
            if i < seq_length:
                continue

            window = df.iloc[i - seq_length + 1: i + 1]
            arr = window[features].values.astype(np.float32)

            # Min-max normalize per window
            mins = arr.min(axis=0, keepdims=True)
            maxs = arr.max(axis=0, keepdims=True)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            normalized = (arr - mins) / ranges

            target = int(df.loc[timestamp]['Target'])
            X.append(normalized)
            y.append(target)
        except Exception:
            continue

    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def walk_forward_backtest(window_months=6, test_months=1, payout=0.80):
    """
    Walk-forward analysis:
      - Train on `window_months` of data
      - Test on the next `test_months`
      - Roll forward by `test_months` and repeat

    This gives the most realistic estimate of live performance.
    """
    # Import here to avoid circular dependency and make it optional
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, BatchNormalization, Dropout,
        Bidirectional, LSTM, Dense, Flatten
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.utils import class_weight

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Ensure indicators
    if 'SMA_20' not in df.columns:
        df = add_all_indicators(df)
    if 'Target' not in df.columns:
        df['Target'], _, _ = create_risk_adjusted_labels(df)

    df['Pattern'] = detect_patterns_with_confirmation(df)
    df.dropna(subset=TABULAR_FEATURES + ['Target'], inplace=True)

    # Time range
    start_date = df.index.min()
    end_date = df.index.max()
    total_days = (end_date - start_date).days

    print(f"Data range: {start_date.date()} -> {end_date.date()} ({total_days} days)")
    print(f"Walk-forward: {window_months}mo train -> {test_months}mo test\n")

    results = []
    fold = 1
    train_start = start_date

    header = (f"{'Fold':<6} | {'Test Period':<25} | {'Win Rate':<10} | "
              f"{'Precision':<10} | {'Trades':<8} | {'P/L':<10} | {'EV/Trade':<10}")
    print(header)
    print("-" * 90)

    while True:
        train_end = train_start + pd.DateOffset(months=window_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        # Split by time
        train_mask = (df.index >= train_start) & (df.index < train_end) & df['Pattern']
        test_mask = (df.index >= train_end) & (df.index < test_end) & df['Pattern']

        train_indices = df[train_mask].index
        test_indices = df[test_mask].index

        if len(train_indices) < 50 or len(test_indices) < 10:
            train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        # Prepare sequences
        X_train, y_train = prepare_sequences(df, train_indices, TABULAR_FEATURES, SEQUENCE_LENGTH)
        X_test, y_test = prepare_sequences(df, test_indices, TABULAR_FEATURES, SEQUENCE_LENGTH)

        if X_train is None or X_test is None:
            train_start += pd.DateOffset(months=test_months)
            fold += 1
            continue

        # Build a lightweight model for each fold (fast training)
        n_features = X_train.shape[2]
        model = Sequential([
            Conv1D(32, 3, padding='same', activation='relu',
                   input_shape=(SEQUENCE_LENGTH, n_features)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy', metrics=['accuracy'])

        # Class weights
        try:
            weights = class_weight.compute_class_weight(
                class_weight='balanced', classes=np.unique(y_train), y=y_train
            )
            w_dict = dict(enumerate(weights))
        except ValueError:
            w_dict = {0: 1.0, 1: 1.0}

        # Train (quiet, fast)
        model.fit(
            X_train, y_train,
            validation_split=0.15,
            batch_size=128, epochs=10,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
            class_weight=w_dict,
            verbose=0
        )

        # Predict
        preds = model.predict(X_test, verbose=0).ravel()
        y_pred = (preds >= CONFIDENCE_THRESHOLD).astype(int)

        trades = y_pred.sum()
        if trades > 0:
            wins = ((y_pred == 1) & (y_test == 1)).sum()
            win_rate = wins / trades
            prec = precision_score(y_test, y_pred, zero_division=0)
            profit = (wins * payout) - ((trades - wins) * 1.0)
            ev = profit / trades
        else:
            win_rate = prec = profit = ev = 0
            wins = 0

        test_period = f"{train_end.date()} -> {test_end.date()}"
        print(f"{fold:<6} | {test_period:<25} | {win_rate:<10.2%} | "
              f"{prec:<10.2%} | {trades:<8d} | {profit:<+10.2f} | {ev:<+10.4f}")

        results.append({
            'fold': fold,
            'test_start': train_end,
            'test_end': test_end,
            'win_rate': win_rate,
            'precision': prec,
            'trades': trades,
            'profit': profit,
            'ev_per_trade': ev
        })

        # Clean up model to free memory
        del model
        tf.keras.backend.clear_session()

        # Roll forward
        train_start += pd.DateOffset(months=test_months)
        fold += 1

    # Summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 90)
        print("  WALK-FORWARD SUMMARY")
        print("=" * 90)
        print(f"Total Folds: {len(results_df)}")
        print(f"Average Win Rate: {results_df['win_rate'].mean():.2%}")
        print(f"Average Precision: {results_df['precision'].mean():.2%}")
        print(f"Total Trades: {results_df['trades'].sum()}")
        print(f"Total Profit: {results_df['profit'].sum():+.2f}")
        print(f"Average EV/Trade: {results_df['ev_per_trade'].mean():+.4f}")

        # Best/worst folds
        if len(results_df) > 1:
            best = results_df.loc[results_df['profit'].idxmax()]
            worst = results_df.loc[results_df['profit'].idxmin()]
            print(f"\nBest Fold: #{int(best['fold'])} ({best['profit']:+.2f})")
            print(f"Worst Fold: #{int(worst['fold'])} ({worst['profit']:+.2f})")

        # Save results
        results_df.to_csv("walkforward_results.csv", index=False)
        print(f"\nDetailed results saved to 'walkforward_results.csv'")
    else:
        print("\nNo folds completed. Need more data or shorter windows.")


if __name__ == "__main__":
    walk_forward_backtest()
