"""
Shared configuration for the Options Trader pipeline.
All scripts import from here to avoid hardcoded inconsistencies.
"""
import numpy as np
import tensorflow as tf
import random
import os

# ============================================================
# REPRODUCIBILITY — Set seeds globally before anything else
# ============================================================
SEED = 42

def set_seeds(seed=SEED):
    """Set all random seeds for reproducible results."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============================================================
# DATA CONFIGURATION
# ============================================================
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
SINCE_DATE = "2024-01-01 00:00:00"
INPUT_FILE = "big_data_btc.csv"

# ============================================================
# LABELING STRATEGY
# ============================================================
LOOK_AHEAD = 15           # Predict N minutes into the future
LABEL_THRESHOLD = 0.003   # 0.3% move required for directional label
PROFIT_TARGET = 0.003     # 0.3% gain for pattern-based labels
REWARD_RISK_RATIO = 1.5   # Reward must exceed risk by this factor

# ============================================================
# IMAGE GENERATION
# ============================================================
IMG_WINDOW = 60            # Number of candles per image
IMG_DIM = (224, 224)       # Increased from 100x100 for better detail
IMG_DPI = 75               # Render quality (higher = slower but clearer)
IMAGE_DATASET_DIR = "dataset_images"

# ============================================================
# TABULAR DATA GENERATION
# ============================================================
TABULAR_DATASET_DIR = "dataset_tabular"
SEQUENCE_LENGTH = 60       # Must match IMG_WINDOW for consistency
TABULAR_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'BB_Upper', 'BB_Lower',
    'RSI_14', 'ATR_14', 'Volume_Ratio',
    'Returns', 'Body_Ratio', 'Upper_Shadow', 'Lower_Shadow'
]

# ============================================================
# MODEL TRAINING
# ============================================================
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8          # 80% train, 20% validation (chronological)

# Model save paths
CNN_MODEL_PATH = "best_cnn_model.keras"
TABULAR_MODEL_PATH = "best_tabular_model.keras"
ENSEMBLE_WEIGHTS = {'cnn': 0.4, 'tabular': 0.6}  # Tabular gets more weight

# ============================================================
# LIVE TRADING BOT
# ============================================================
CONFIDENCE_THRESHOLD = 0.60
VOLUME_CONFIRMATION_MULTIPLIER = 1.5  # Volume must be 1.5x avg for pattern
ATR_VOLATILITY_PERIODS = 14
MIN_ATR_PERCENTILE = 25   # Skip trading when ATR is in bottom 25% of recent history

# ============================================================
# INDICATOR PARAMETERS
# ============================================================
SMA_PERIOD = 20
BB_PERIOD = 20
BB_STD_DEV = 2
RSI_PERIOD = 14
ATR_PERIOD = 14
