# utils.py
"""
Utility functions for the project, including:
- Setting random seeds for reproducibility
- Saving and loading model checkpoints
- Plotting training history and prediction results
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, path):
    """
    Saves the model's state dictionary to a file.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save to.
    """
    print(f"Saving model checkpoint to {path}")
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    """
    Loads a model's state dictionary from a file.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): The file path to load from.
        device (torch.device): The device to map the model to.

    Returns:
        torch.nn.Module: The model with loaded weights, set to eval mode.
    """
    print(f"Loading model checkpoint from {path}...")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode after loading
    return model


def plot_training_history(history, save_path):
    """
    Plots the training and validation loss and MAE curves.

    Args:
        history (dict): A dictionary containing 'loss', 'val_loss', 'mae', 'val_mae'.
        save_path (str): The file path to save the plot image.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss (MSE)
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Metric (MAE)
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE (SpO2 %)')
    plt.plot(history['val_mae'], label='Validation MAE (SpO2 %)')
    plt.title('Model Mean Absolute Error (SpO2 %)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nTraining history plot saved to {save_path}")


def plot_prediction_results(df, save_path, title='Prediction vs Actual'):
    """
    Plots the predicted values against the actual (true) values.

    Args:
        df (pd.DataFrame): DataFrame containing 'average_SpO2' (optional)
                           and 'predicted_SpO2'.
        save_path (str): The file path to save the plot image.
        title (str): The title for the plot.
    """
    if df.empty:
        print("DataFrame is empty, skipping plot generation.")
        return

    plt.figure(figsize=(15, 7))

    # Plot Actual Values (if they exist in the dataframe)
    if 'average_SpO2' in df.columns:
        plt.plot(df.index, df['average_SpO2'], label='Actual SpO2', color='blue', alpha=0.8, linewidth=1)

    # Plot Predicted Values
    plt.plot(df.index, df['predicted_SpO2'], label='Predicted SpO2', color='red', linestyle='--', alpha=0.8,
             linewidth=1)

    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('SpO2 (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(60, 105)  # Set Y-axis limits for better SpO2 visualization

    plt.savefig(save_path)
    print(f"\nPrediction plot saved to: {save_path}")
