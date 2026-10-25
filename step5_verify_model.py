"""
Step 5: Model verification with comprehensive evaluation metrics.

Changes from original:
  - Evaluates both CNN and Tabular models
  - Walk-forward style evaluation on chronological validation set
  - Adds Precision, Recall, F1, ROC-AUC, and profit simulation
  - Confusion matrix for each model
  - Ensemble evaluation
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    IMAGE_DATASET_DIR, TABULAR_DATASET_DIR, IMG_DIM, BATCH_SIZE,
    CNN_MODEL_PATH, TABULAR_MODEL_PATH, ENSEMBLE_WEIGHTS,
    CONFIDENCE_THRESHOLD, TRAIN_SPLIT, SEED, set_seeds
)
from utils import get_chronological_split
from step3_train_tabular import SelfAttention

set_seeds(SEED)


def evaluate_model(y_true, y_pred_proba, threshold, model_name):
    """Print comprehensive evaluation metrics for a single model."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Threshold: {threshold:.2f}")
    print(f"{'=' * 50}")

    print(classification_report(
        y_true, y_pred,
        target_names=['Wait (0)', 'Buy (1)'],
        zero_division=0
    ))

    # ROC-AUC (only if both classes present)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        print("ROC-AUC: N/A (single class in validation)")

    # Trade statistics
    trades = y_pred.sum()
    if trades > 0:
        wins = ((y_pred == 1) & (y_true == 1)).sum()
        win_rate = wins / trades
        print(f"\n--- Trading Statistics ---")
        print(f"Signals Generated: {trades}")
        print(f"Profitable Signals: {wins}")
        print(f"Win Rate: {win_rate:.2%}")

        # Simple P/L simulation (binary options: win = +80%, loss = -100%)
        payout = 0.80
        profit = (wins * payout) - ((trades - wins) * 1.0)
        print(f"Simulated P/L (80% payout): {profit:+.2f} units")
        print(f"Expected Value per trade: {profit / trades:+.4f}")
    else:
        print("No trades generated at this threshold.")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Wait', 'Pred Buy'],
                yticklabels=['True Wait', 'True Buy'], ax=ax)
    ax.set_title(f'{model_name} — Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    fname = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"\nConfusion matrix saved: {fname}")

    return y_pred_proba


def verify_cnn():
    """Evaluate the CNN model on chronological validation data."""
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"CNN model not found: {CNN_MODEL_PATH}")
        return None, None

    print("\n--- Evaluating CNN Model ---")
    train_df, val_df, _ = get_chronological_split(IMAGE_DATASET_DIR, split_ratio=TRAIN_SPLIT)
    if val_df is None or len(val_df) == 0:
        print("No validation data for CNN.")
        return None, None

    datagen = ImageDataGenerator(rescale=1. / 255)
    val_gen = datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=IMG_DIM, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    preds = model.predict(val_gen, verbose=1).ravel()
    y_true = val_gen.classes

    evaluate_model(y_true, preds, CONFIDENCE_THRESHOLD, "CNN (EfficientNetB0)")
    return y_true, preds


def verify_tabular():
    """Evaluate the tabular model on chronological validation data."""
    if not os.path.exists(TABULAR_MODEL_PATH):
        print(f"Tabular model not found: {TABULAR_MODEL_PATH}")
        return None, None

    val_path = os.path.join(TABULAR_DATASET_DIR, "val.npz")
    if not os.path.exists(val_path):
        print(f"Validation data not found: {val_path}")
        return None, None

    print("\n--- Evaluating Tabular Model ---")
    val_data = np.load(val_path)
    X_val, y_val = val_data['X'], val_data['y']

    model = tf.keras.models.load_model(TABULAR_MODEL_PATH, custom_objects={'SelfAttention': SelfAttention})
    preds = model.predict(X_val, verbose=1).ravel()

    evaluate_model(y_val, preds, CONFIDENCE_THRESHOLD, "Tabular (CNN+BiLSTM)")
    return y_val, preds


def verify_ensemble(cnn_result, tabular_result):
    """Evaluate the weighted ensemble of both models."""
    if cnn_result[0] is None and tabular_result[0] is None:
        print("No models to ensemble.")
        return

    # If both models are available AND share the same validation set size
    if (cnn_result[0] is not None and tabular_result[0] is not None
            and len(cnn_result[0]) == len(tabular_result[0])):
        print("\n--- Evaluating Ensemble ---")
        y_true = cnn_result[0]
        cnn_preds = cnn_result[1]
        tab_preds = tabular_result[1]

        w_cnn = ENSEMBLE_WEIGHTS['cnn']
        w_tab = ENSEMBLE_WEIGHTS['tabular']
        total_w = w_cnn + w_tab

        ensemble_preds = (cnn_preds * w_cnn + tab_preds * w_tab) / total_w
        evaluate_model(y_true, ensemble_preds, CONFIDENCE_THRESHOLD, "Ensemble")
    else:
        print("\nSkipping ensemble (models have different validation sets).")


if __name__ == "__main__":
    print("=" * 60)
    print("  MODEL VERIFICATION SUITE")
    print("=" * 60)

    cnn_result = verify_cnn()
    tabular_result = verify_tabular()
    verify_ensemble(cnn_result, tabular_result)

    print("\n[Done] Verification complete.")