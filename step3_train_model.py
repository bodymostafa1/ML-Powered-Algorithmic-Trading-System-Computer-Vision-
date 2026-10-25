"""
Step 3A: Train CNN model on candlestick chart images.

Changes from original:
  - Uses shared config, utils, and chronological split
  - EfficientNetB0 instead of generic CNN (6x fewer params than ResNet50)
  - Fine-tunes last 20 layers instead of freezing everything
  - Proper data augmentation (horizontal flip for chart symmetry)
  - Seeds set for reproducibility
  - Monitors val_loss with patience-based early stopping
  - Class weighting to handle imbalance
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

from config import (
    IMAGE_DATASET_DIR, IMG_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CNN_MODEL_PATH, TRAIN_SPLIT, SEED, set_seeds
)
from utils import get_chronological_split

set_seeds(SEED)


def build_model():
    """
    EfficientNetB0 with fine-tuned last 20 layers.
    Better than ResNet50 for small datasets: fewer parameters, better accuracy.
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_DIM[0], IMG_DIM[1], 3)
    )

    # Freeze all layers EXCEPT the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE * 0.1),  # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    trainable_count = sum(1 for l in model.layers if l.trainable)
    total_count = len(model.layers)
    print(f"Model: EfficientNetB0 | Trainable layers: {trainable_count}/{total_count}")
    print(f"Total params: {model.count_params():,}")

    return model


def train_cnn():
    """Train CNN with chronological split, class weighting, and data augmentation."""
    # Get chronological split with class weights
    train_df, val_df, class_weights = get_chronological_split(
        IMAGE_DATASET_DIR, split_ratio=TRAIN_SPLIT
    )
    if train_df is None:
        print("Failed to load dataset. Exiting.")
        return

    # Data augmentation for training (horizontal flip makes sense for charts)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,   # Mirror images — chart patterns are symmetric
        width_shift_range=0.05, # Slight horizontal jitter
        zoom_range=0.05,        # Minimal zoom
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, x_col='filename', y_col='class',
        target_size=IMG_DIM, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True
    )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, x_col='filename', y_col='class',
        target_size=IMG_DIM, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    print(f"\n--- Building EfficientNetB0 Model ---")
    model = build_model()

    callbacks = [
        ModelCheckpoint(
            CNN_MODEL_PATH,
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

    print("\n--- Starting Weighted Training ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print(f"\n[Done] Best model saved to '{CNN_MODEL_PATH}'")
    return history


if __name__ == "__main__":
    train_cnn()