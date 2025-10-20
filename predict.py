# predict.py
"""
Prediction (Inference) Script.

This script loads a pre-trained model and a saved Scaler
to make predictions on a new, unseen CSV file.
"""

import os
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from torch.utils.data import DataLoader

# Local module imports
import config
from models import get_model_class
from data_utils import PredictionDataset
from utils import load_checkpoint, plot_prediction_results


def predict_spo2(csv_path, scaler_path, checkpoint_path):
    """
    Loads data from a single CSV, preprocesses it, and runs inference
    using the specified model and scaler paths.

    Args:
        csv_path (str): Path to the input CSV file (must contain 'R', 'G', 'B').
        scaler_path (str): Path to the scaler.pkl file for this fold.
        checkpoint_path (str): Path to the best.pth file for this fold.

    Returns:
        pd.DataFrame: A DataFrame with original data + 'predicted_SpO2' column.
    """
    device = config.DEVICE
    print(f"Using device: {device}")
    print(f"--- Using Model: {config.MODEL_NAME} ---")
    print(f"--- Model Path: {checkpoint_path} ---")
    print(f"--- Scaler Path: {scaler_path} ---")

    # --- 1. Check for Required Files ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'.")
        return pd.DataFrame()

    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at '{scaler_path}'.")
        return pd.DataFrame()

    if not os.path.exists(csv_path):
        print(f"Error: Input CSV file not found at '{csv_path}'.")
        return pd.DataFrame()

    # --- 2. Load Scaler ---
    print(f"Loading Scaler from '{scaler_path}'...")
    scaler = joblib.load(scaler_path)

    # --- 3. Load Model ---
    print(f"Initializing model architecture: {config.MODEL_NAME}...")
    try:
        ModelClass = get_model_class(config.MODEL_NAME)
        model = ModelClass(**config.MODEL_ARGS)
    except Exception as e:
        print(f"Failed to initialize model {config.MODEL_NAME} with args {config.MODEL_ARGS}")
        print(f"Error: {e}")
        return pd.DataFrame()

    print(f"Loading model weights from '{checkpoint_path}'...")
    model = load_checkpoint(model, checkpoint_path, device)

    # --- 4. Load and Preprocess Prediction Data ---
    print(f"\nLoading data to predict from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    if len(df) < config.SEQUENCE_LENGTH:
        print(f"Error: Data has {len(df)} rows, but {config.SEQUENCE_LENGTH} are required for one sequence.")
        return pd.DataFrame()

    features_to_predict = df[['R', 'G', 'B']].values
    features_scaled = scaler.transform(features_to_predict)
    print("Data loading and standardization complete.")

    # --- 5. Create DataLoader and Run Prediction ---
    prediction_dataset = PredictionDataset(features_scaled, config.SEQUENCE_LENGTH)
    prediction_loader = DataLoader(prediction_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print("\nRunning prediction...")
    all_predictions = []
    with torch.no_grad():
        for seq_batch in tqdm(prediction_loader, desc="Predicting"):
            seq_batch = seq_batch.to(device)
            outputs = model(seq_batch)
            all_predictions.extend(outputs.cpu().numpy().flatten())
    print("Prediction complete.")

    # --- 6. Format Results ---
    # Predictions correspond to the center of the sequence
    center_offset = config.SEQUENCE_LENGTH // 2
    num_predictions = len(all_predictions)

    # Select the corresponding rows from the original dataframe
    result_df = df.iloc[center_offset: center_offset + num_predictions].copy()

    # Add the predictions as a new column
    result_df['predicted_SpO2'] = all_predictions

    # Define columns to return (include actual SpO2 if present)
    columns_to_return = ['帧序号', '视频时间', 'average_SpO2', 'predicted_SpO2']
    # Filter for columns that actually exist in the dataframe
    final_columns = [col for col in columns_to_return if col in result_df.columns]

    return result_df[final_columns]


def main(csv_to_predict):
    """
    Main function to run prediction and plot results.
    Loads paths based on the fold configuration below.

    Args:
        csv_to_predict (str): Path to the input CSV file.
    """

    # --- Configuration ---
    # Set to True if you trained with RUN_LEAVE_ONE_OUT_CV = True
    CV_MODE_FOR_PREDICTION = True

    # **IMPORTANT**: Set these to the indices of the fold you want to load
    PREDICTION_TEST_IDX = 0
    PREDICTION_VAL_IDX = 0
    # ---------------------

    # Determine which paths to use
    if CV_MODE_FOR_PREDICTION:
        print(f"--- Loading fold: Test={PREDICTION_TEST_IDX}, Val={PREDICTION_VAL_IDX} ---")
        scaler_path, model_results_dir, checkpoint_path = config.get_cv_paths(
            config.MODEL_NAME, PREDICTION_TEST_IDX, PREDICTION_VAL_IDX
        )
    else:
        print("--- Loading from Random Mode (default) paths ---")
        scaler_path = config.DEFAULT_SCALER_PATH
        model_results_dir = config.DEFAULT_MODEL_RESULTS_DIR
        checkpoint_path = config.DEFAULT_CHECKPOINT_PATH

    # Run the prediction
    prediction_results_df = predict_spo2(
        csv_path=csv_to_predict,
        scaler_path=scaler_path,
        checkpoint_path=checkpoint_path
    )

    if not prediction_results_df.empty:
        print("\n--- Prediction Results (Head) ---")
        print(prediction_results_df.head())
        print("---------------------------------\n")

        # Plot the results
        plot_prediction_results(
            prediction_results_df,
            save_path=f"{model_results_dir}/prediction_vs_actual.png",
            title=f"Prediction vs Actual ({config.MODEL_NAME}) for {os.path.basename(csv_to_predict)}"
        )


if __name__ == '__main__':
    # --- Configuration ---
    # Specify the CSV file you want to run predictions on.
    CSV_TO_PREDICT = r"C:\Users\26788\Desktop\pig_project\processed_output\100001_Left.csv"

    main(csv_to_predict=CSV_TO_PREDICT)