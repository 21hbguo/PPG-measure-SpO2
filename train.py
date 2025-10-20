# train.py
"""
Main Training Script.

This script orchestrates the entire training and evaluation process.
It can run in two modes based on config.RUN_LEAVE_ONE_OUT_CV:
1. CV Mode (True): Iterates through all 4:1:1 subject splits.
2. Random Mode (False): Runs a single random 4:1:1 split.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import pandas as pd

# Local module imports
import config
from models import get_model_class
# Import new helper function
from data_utils import get_all_subjects, get_or_create_scaler, get_data_loaders
from utils import set_seed, save_checkpoint, load_checkpoint, plot_training_history, plot_prediction_results


def train_one_epoch(model, loader, optimizer, criterion, metric, device, clip_norm):
    """
    Executes a single training epoch.

    Returns:
        tuple: (average epoch loss, average epoch MAE)
    """
    model.train()
    running_loss, running_mae = 0.0, 0.0

    progress_bar = tqdm(loader, desc=f"Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        # Update running metrics
        mae_val = metric(outputs, labels).item()
        running_loss += loss.item()
        running_mae += mae_val
        progress_bar.set_postfix(loss=loss.item(), mae=mae_val)

    epoch_loss = running_loss / len(loader)
    epoch_mae = running_mae / len(loader)
    return epoch_loss, epoch_mae


def evaluate(model, loader, criterion, metric, device, desc="Validating"):
    """
    Evaluates the model on a given dataset (validation or test).

    Returns:
        tuple: (avg_loss, avg_mae, predictions_array, true_values_array)
    """
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            mae_val = metric(outputs, labels)

            total_loss += loss.item()
            total_mae += mae_val.item()
            progress_bar.set_postfix(loss=loss.item(), mae=mae_val.item())

            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)

    # Flatten all batch results into single arrays
    predictions = np.concatenate([p.flatten() for p in all_predictions])
    true_values = np.concatenate([l.flatten() for l in all_labels])

    return avg_loss, avg_mae, predictions, true_values


def run_training_fold(config, all_subjects, test_idx, val_idx):
    """
    Runs the complete training and evaluation workflow for a single fold.
    """

    # --- 0. Setup Fold ---
    device = config.DEVICE

    # Get dynamic paths for this fold
    if test_idx is not None:
        # CV Mode: Use dynamic paths
        scaler_path, model_results_dir, checkpoint_path = config.get_cv_paths(
            config.MODEL_NAME, test_idx, val_idx
        )
    else:
        # Random Mode: Use default paths
        scaler_path = config.DEFAULT_SCALER_PATH
        model_results_dir = config.DEFAULT_MODEL_RESULTS_DIR
        checkpoint_path = config.DEFAULT_CHECKPOINT_PATH

    # Create model-specific results subdirectory for this fold
    os.makedirs(model_results_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"--- Running Model: {config.MODEL_NAME} ---")
    print(f"Results will be saved to: {model_results_dir}")

    # --- 1. Get Scaler ---
    # Load or create the StandardScaler for this *specific training split*
    scaler = get_or_create_scaler(config, scaler_path, all_subjects, test_idx, val_idx)

    # --- 2. Get DataLoaders ---
    try:
        # Get DataLoaders for this *specific training split*
        train_loader, val_loader, test_loader = get_data_loaders(config, scaler, all_subjects, test_idx, val_idx)
    except ValueError as e:
        print(e)
        return

    # --- 3. Initialize Model and Optimizers ---
    print(f"\nInitializing model: {config.MODEL_NAME}")
    try:
        ModelClass = get_model_class(config.MODEL_NAME)
        model = ModelClass(**config.MODEL_ARGS).to(device)
    except Exception as e:
        print(f"Failed to initialize model {config.MODEL_NAME} with args {config.MODEL_ARGS}")
        print(f"Error: {e}")
        return

    criterion = nn.MSELoss()  # Loss function: Mean Squared Error
    metric_mae = nn.L1Loss()  # Evaluation metric: Mean Absolute Error

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)

    print("\nModel Architecture:")
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- 4. Training Loop ---
    print("\nStarting model training...")
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")

        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, metric_mae, device, config.GRAD_CLIP_MAX_NORM
        )
        val_loss, val_mae, _, _ = evaluate(
            model, val_loader, criterion, metric_mae, device, desc="Validating"
        )

        scheduler.step()

        # Log history
        history['loss'].append(train_loss)
        history['mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.4f} | "
            f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.4f}"
        )

        # Check for improvement and save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, checkpoint_path)
        else:
            epochs_no_improve += 1

        # Check for early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered: No improvement in {config.EARLY_STOPPING_PATIENCE} epochs.")
            break

    print("\nTraining complete.")

    # --- 5. Final Evaluation ---
    print(f"\nLoading best model from {checkpoint_path} for final evaluation...")
    # Re-initialize a clean model and load the best weights
    ModelClass = get_model_class(config.MODEL_NAME)
    model = ModelClass(**config.MODEL_ARGS)
    model = load_checkpoint(model, checkpoint_path, device)

    # Evaluate on Validation Set (with plotting)
    val_loss, val_mae, val_preds, val_true = evaluate(
        model, val_loader, criterion, metric_mae, device, desc="Final Validation"
    )
    print(f"Best Model (Validation) - Loss (MSE): {val_loss:.6f}, MAE: {val_mae:.4f}")

    val_df = pd.DataFrame({'average_SpO2': val_true, 'predicted_SpO2': val_preds})
    plot_prediction_results(
        val_df,
        save_path=f"{model_results_dir}/validation_performance.png",
        title=f"Best Model ({config.MODEL_NAME}) Performance on Validation Set"
    )

    # Evaluate on Test Set
    test_loss, test_mae, _, _ = evaluate(
        model, test_loader, criterion, metric_mae, device, desc="Testing"
    )
    print(f"Best Model (Test Set) - Loss (MSE): {test_loss:.6f}, MAE: {test_mae:.4f}")

    # --- 6. Plot Training History ---
    plot_training_history(history, save_path=f"{model_results_dir}/training_history.png")


def main():
    """Main training controller."""

    # --- 0. Global Setup ---
    set_seed(config.RANDOM_SEED)
    # Create base results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Get all available subjects *before* starting any loops
    all_subjects = get_all_subjects(config)
    num_subjects = len(all_subjects)

    if num_subjects == 0:
        print("Error: No processed data files found in 'data/processed'. Please run preprocess_data.py first.")
        return

    print(f"Found {num_subjects} total subjects: {all_subjects}")

    # --- Check Mode ---
    if config.RUN_LEAVE_ONE_OUT_CV:
        # --- Run Full CV Mode ---
        print(f"\n{'=' * 60}\nSTARTING AUTOMATED LEAVE-ONE-OUT CROSS-VALIDATION\n{'=' * 60}")
        if num_subjects < 3:
            print(f"Error: Need at least 3 subjects for a Train/Val/Test split. Found {num_subjects}.")
            return

        total_folds = num_subjects * (num_subjects - 1)
        current_fold = 1

        for i in range(num_subjects):  # Outer loop: Test Subject Index
            for j in range(num_subjects):  # Inner loop: Validation Subject Index
                if i == j:
                    continue  # Skip fold where test == val

                print(f"\n\n{'=' * 60}")
                print(f"STARTING CV FOLD {current_fold}/{total_folds}: "
                      f"Test Subject: {all_subjects[i]} (idx {i}), "
                      f"Val Subject: {all_subjects[j]} (idx {j})")
                print(f"{'=' * 60}")

                run_training_fold(
                    config=config,
                    all_subjects=all_subjects,
                    test_idx=i,
                    val_idx=j
                )
                current_fold += 1

        print(f"\n\n{'=' * 60}\nAUTOMATED CROSS-VALIDATION COMPLETE\n{'=' * 60}")

    else:
        # --- Run Single Random Mode ---
        print(f"\n\n{'=' * 60}\nSTARTING SINGLE RANDOM SPLIT (4:1:1) MODE\n{'=' * 60}")
        if num_subjects < 6:
            print(f"Warning: Running random split, but found fewer than 6 subjects ({num_subjects}).")

        run_training_fold(
            config=config,
            all_subjects=all_subjects,
            test_idx=None,  # Pass None to trigger random split
            val_idx=None  # Pass None to trigger random split
        )
        print(f"\n\n{'=' * 60}\nSINGLE RANDOM SPLIT COMPLETE\n{'=' * 60}")


if __name__ == '__main__':
    main()