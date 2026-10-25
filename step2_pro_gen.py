"""
Step 2A: Generate candlestick chart images for CNN training.

Consolidated from 4 separate generators (image_gen, elite_gen, pattern_gen, rsi_gen, pro_gen).

Changes from original:
  - Uses shared config & utils
  - Volume-confirmed pattern detection (not just any engulfing)
  - Risk-adjusted labels from CSV (not recalculated here)
  - Higher resolution images (224x224) for better indicator visibility
  - Includes volume panel + RSI panel in every image
  - Consistent indicator rendering between training and live inference
"""
import pandas as pd
import mplfinance as mpf
import os
import matplotlib.pyplot as plt
import numpy as np

from config import (
    INPUT_FILE, IMAGE_DATASET_DIR, IMG_WINDOW, IMG_DIM, IMG_DPI,
    VOLUME_CONFIRMATION_MULTIPLIER, SEED, set_seeds
)
from utils import (
    add_all_indicators, detect_patterns_with_confirmation, calculate_rsi
)

set_seeds(SEED)


def create_folders(output_dir):
    """Create dataset directory structure."""
    for label in ['0', '1']:
        path = os.path.join(output_dir, label)
        os.makedirs(path, exist_ok=True)


def get_plot_style():
    """Return the standardized mplfinance style used for ALL image generation."""
    mc = mpf.make_marketcolors(
        up='#00ff00', down='#ff0000',
        edge='inherit', wick='inherit', volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle='',
        rc={
            'axes.axisbelow': True,
            'figure.facecolor': 'black',
            'axes.facecolor': 'black',
            'axes.edgecolor': 'black'
        }
    )
    return style


def generate_single_image(window, filename, style):
    """
    Render a single candlestick image with indicators.
    Returns True on success, False on failure.
    """
    try:
        # Build additional plots: SMA, Bollinger Bands on price panel
        apds = [
            mpf.make_addplot(window['SMA_20'], color='blue', width=1.5, panel=0),
            mpf.make_addplot(window['BB_Upper'], color='white', width=0.5, linestyle='dotted', panel=0),
            mpf.make_addplot(window['BB_Lower'], color='white', width=0.5, linestyle='dotted', panel=0),
        ]

        # Add RSI panel if data is available
        if 'RSI_14' in window.columns and window['RSI_14'].notna().any():
            apds.append(
                mpf.make_addplot(window['RSI_14'], color='purple', width=1.0, panel=2, ylim=(0, 100))
            )

        mpf.plot(
            window, type='candle', style=style,
            addplot=apds,
            volume=True,
            axisoff=True,
            savefig=dict(fname=filename, dpi=IMG_DPI, bbox_inches='tight', pad_inches=0),
            returnfig=True, closefig=True,
            panel_ratios=(3, 1, 1)  # Price, Volume, RSI
        )
        return True

    except Exception as e:
        # Silently skip rendering failures (e.g., NaN in window)
        return False


def generate_images():
    """Main generation loop. Only generates images where patterns are detected."""
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Ensure indicators exist (they should from step1, but recalculate if needed)
    if 'SMA_20' not in df.columns:
        print("Indicators not found in CSV, recalculating...")
        df = add_all_indicators(df)
        print("Saving indicators back to CSV for future runs...")
        df.to_csv(INPUT_FILE)
        print("Saved.")

    # Detect patterns with volume confirmation
    df['Pattern'] = detect_patterns_with_confirmation(
        df, volume_multiplier=VOLUME_CONFIRMATION_MULTIPLIER
    )

    # The 'Target' column should already exist from step1 (risk-adjusted)
    if 'Target' not in df.columns:
        print("ERROR: 'Target' column not found. Run step1_data_prep.py first.")
        return

    df.dropna(inplace=True)

    # Only generate images at pattern locations
    pattern_indices = df[df['Pattern']].index
    print(f"Found {len(pattern_indices)} volume-confirmed pattern setups.")
    print(f"Generating {IMG_DIM[0]}x{IMG_DIM[1]} images with Volume + RSI panels...")

    style = get_plot_style()
    plt.switch_backend('Agg')

    count = 0
    skipped = 0

    for timestamp in pattern_indices:
        try:
            i = df.index.get_loc(timestamp)
            if i < IMG_WINDOW:
                continue

            window = df.iloc[i - IMG_WINDOW + 1: i + 1]
            target = int(df.loc[timestamp]['Target'])

            filename = os.path.join(IMAGE_DATASET_DIR, str(target), f"img_{count}.png")

            if generate_single_image(window, filename, style):
                count += 1
            else:
                skipped += 1

            if count % 500 == 0 and count > 0:
                print(f"Generated {count} images (skipped {skipped})...", end='\r')

        except Exception as e:
            skipped += 1
            continue

    plt.close('all')

    # Report class distribution
    class_0 = len(os.listdir(os.path.join(IMAGE_DATASET_DIR, '0')))
    class_1 = len(os.listdir(os.path.join(IMAGE_DATASET_DIR, '1')))
    print(f"\n[Done] Generated {count} images in '{IMAGE_DATASET_DIR}' (skipped {skipped})")
    print(f"Class Distribution — 0 (Wait): {class_0} | 1 (Buy): {class_1}")
    if class_0 + class_1 > 0:
        print(f"Buy Ratio: {class_1 / (class_0 + class_1):.2%}")


if __name__ == "__main__":
    create_folders(IMAGE_DATASET_DIR)
    generate_images()