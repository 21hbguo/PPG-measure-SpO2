# config.py
"""
Project Configuration File.
Contains all paths, hyperparameters, and settings for the project.
"""

import torch
import os

# --- 1. Path Configuration ---
# Get the absolute path of the directory where config.py is located
# (This is the project's root directory)
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

# Input paths (for preprocessing)
VIDEO_ROOT_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")
GT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "gt")

# Output from preprocess_data.py, Input for train.py
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# --- 2. Data Parameters ---
SEQUENCE_LENGTH = 90  # Frame sequence length
INPUT_CHANNELS = 3    # Input features (R, G, B)
OUTPUT_SIZE = 1       # Output features (SpO2 value)


# --- 3. Training Hyperparameters ---
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.1      # L2 regularization
DROPOUT_RATE = 0.7      # Default dropout rate
RANDOM_SEED = 42        # Seed for reproducibility


# --- 4. Training Control ---
EARLY_STOPPING_PATIENCE = 15  # Epochs to wait for improvement
SCHEDULER_STEP_SIZE = 80      # Learning rate scheduler step
SCHEDULER_GAMMA = 0.1         # Learning rate scheduler gamma
GRAD_CLIP_MAX_NORM = 1.0      # Gradient clipping max norm

# --- NEW: Automated Cross-Validation Control ---
# Set to True to automatically run Leave-One-Out (LOO) CV.
# This will iterate through all possible 4:1:1 subject splits.
# Set to False to run a single, random 4:1:1 split (based on RANDOM_SEED).
RUN_LEAVE_ONE_OUT_CV = True  # <--- 这是您的“一键开关”


# --- 5. Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 6. Model Configuration ---

# 6.1. Select Model to Run
# Change this string to "CNN" or "LSTM" to switch models.
MODEL_NAME = "CNN"

# 6.2. Model-Specific Initialization Arguments
MODEL_CONFIGS = {
    "CNN": {
        "sequence_length": SEQUENCE_LENGTH,
        "output_size": OUTPUT_SIZE,
        "dropout_rate": DROPOUT_RATE
    },
    # Renamed "LSTMModel" to "LSTM"
    "LSTM": {
        "input_size": INPUT_CHANNELS,
        "hidden_size": 64,
        "output_size": OUTPUT_SIZE,
        "num_layers": 2,
        "dropout_rate": DROPOUT_RATE
    }
    # Add configurations for new models here
    # "Transformer": { ... }
}

# 6.3. Get arguments for the selected model
try:
    MODEL_ARGS = MODEL_CONFIGS[MODEL_NAME]
except KeyError:
    raise ValueError(f"Configuration for model '{MODEL_NAME}' not found in MODEL_CONFIGS.")


# --- 7. Dynamic Output Paths ---

# These are the *default* paths used for "Random Mode" (CV_MODE = False)
DEFAULT_SCALER_PATH = os.path.join(RESULTS_DIR, "scaler.pkl")
DEFAULT_MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, MODEL_NAME)
DEFAULT_CHECKPOINT_PATH = os.path.join(DEFAULT_MODEL_RESULTS_DIR, "best.pth")

def get_cv_paths(model_name, test_idx, val_idx):
    """
    Generates dynamic paths for a specific CV fold.
    Called by train.py and predict.py.
    """
    fold_suffix = f"fold_test_{test_idx}_val_{val_idx}"
    scaler_path = os.path.join(RESULTS_DIR, f"scaler_test_{test_idx}_val_{val_idx}.pkl")
    model_results_dir = os.path.join(RESULTS_DIR, model_name, fold_suffix)
    checkpoint_path = os.path.join(model_results_dir, "best.pth")
    return scaler_path, model_results_dir, checkpoint_path