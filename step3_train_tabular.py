"""
Step 3B: Train tabular model (1D-CNN + BiLSTM with Attention) on numeric sequences.

This is the primary model — it receives exact floating-point features
instead of lossy pixel approximations from rendered chart images.

Architecture:
  - 1D Convolutional layers extract local patterns (1-5 candle motifs)
  - Bidirectional LSTM captures temporal dependencies
  - Attention layer focuses on the most important candles in the sequence
  - Dense head for binary classification
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout,
    Bidirectional, LSTM, Dense, Attention, GlobalAveragePooling1D,
    Concatenate, Flatten, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
import numpy as np
import os

from config import (
    TABULAR_DATASET_DIR, SEQUENCE_LENGTH, TABULAR_FEATURES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, TABULAR_MODEL_PATH, SEED, set_seeds
)

set_seeds(SEED)


class SelfAttention(Layer):
    """Simple self-attention: learns which timesteps matter most."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, timesteps, features)
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, timesteps, 1)
        weights = tf.nn.softmax(score, axis=1)               # (batch, timesteps, 1)
        context = tf.reduce_sum(x * weights, axis=1)         # (batch, features)
        return context


def build_tabular_model(seq_length, n_features):
    """
    Hybrid 1D-CNN + BiLSTM + Self-Attention model.

    The CNN extracts local patterns (engulfing, hammer, etc.),
    the LSTM models temporal context, and attention focuses on key candles.
    """
    inputs = Input(shape=(seq_length, n_features), name='sequence_input')

    # --- 1D-CNN Branch: Local pattern extraction ---
    c = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    c = BatchNormalization()(c)
    c = Conv1D(128, kernel_size=5, padding='same', activation='relu')(c)
    c = BatchNormalization()(c)
    c = MaxPooling1D(pool_size=2)(c)
    c = Dropout(0.3)(c)

    # --- BiLSTM Branch: Temporal dependencies ---
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(c)
    lstm_out = Dropout(0.3)(lstm_out)

    # --- Self-Attention: Focus on important candles ---
    attention_out = SelfAttention(name='self_attention')(lstm_out)

    # --- Dense Head ---
    x = Dense(128, activation='relu')(attention_out)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid', name='prediction')(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_tabular():
    """Load .npz data and train the tabular model."""
    # Load pre-split data
    train_path = os.path.join(TABULAR_DATASET_DIR, "train.npz")
    val_path = os.path.join(TABULAR_DATASET_DIR, "val.npz")

    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run step2_tabular_gen.py first.")
        return

    print("--- Loading Tabular Data ---")
    train_data = np.load(train_path)
    val_data = np.load(val_path)

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"Train class dist — 0: {(y_train == 0).sum()} | 1: {(y_train == 1).sum()}")

    # Class weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weight_dict = dict(enumerate(weights))
    print(f"Class Weights — 0: {weight_dict[0]:.3f} | 1: {weight_dict[1]:.3f}")

    # Build model
    n_features = X_train.shape[2]
    print(f"\n--- Building 1D-CNN + BiLSTM + Attention Model ---")
    print(f"Sequence length: {SEQUENCE_LENGTH} | Features: {n_features}")

    model = build_tabular_model(SEQUENCE_LENGTH, n_features)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            TABULAR_MODEL_PATH,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n--- Starting Training ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=weight_dict
    )

    print(f"\n[Done] Best model saved to '{TABULAR_MODEL_PATH}'")
    return history


if __name__ == "__main__":
    train_tabular()
