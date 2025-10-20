# data_utils.py
"""
Contains data loading utilities, PyTorch Dataset classes,
and data processing functions (e.g., Scaler handling).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
# Import config directly to be used in load_and_split_data
import config as global_config


# --- 1. PyTorch Dataset Classes ---
# (... PPGDataset 和 PredictionDataset 类保持不变 ...)
class PPGDataset(Dataset):
    """
    PyTorch Dataset for training and validation.
    Generates sequences from continuous data.
    """

    def __init__(self, features, labels, sequence_length):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        # Returns the total number of valid sequences that can be extracted
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Extract a sequence of features
        feature_sequence = self.features[idx:idx + self.sequence_length]

        # Get the label corresponding to the center of the sequence
        label_center = self.labels[idx + self.sequence_length // 2]

        return torch.tensor(feature_sequence, dtype=torch.float32), torch.tensor(label_center, dtype=torch.float32)


class PredictionDataset(Dataset):
    """
    PyTorch Dataset for prediction (inference).
    Generates sequences but does not return labels.
    """

    def __init__(self, features, sequence_length):
        self.features = features
        self.sequence_length = sequence_length

    def __len__(self):
        # Returns the total number of valid sequences
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Extract a sequence of features
        feature_sequence = self.features[idx:idx + self.sequence_length]
        return torch.tensor(feature_sequence, dtype=torch.float32)


# --- 2. Data File Loading and Splitting ---

def load_data_from_files(file_paths):
    """
    Loads and concatenates data from a list of CSV file paths.

    Args:
        file_paths (list): List of paths to CSV files.

    Returns:
        tuple: (features (np.array), labels (np.array))
    """
    if not file_paths:
        return np.array([]), np.array([])

    all_dfs = [pd.read_csv(f) for f in file_paths]
    combined_df = pd.concat(all_dfs, ignore_index=True)

    if combined_df.isnull().values.any():
        print("Warning: NaN values detected in data.")

    feature_columns = ['R', 'G', 'B']
    label_column = 'average_SpO2'

    features = combined_df[feature_columns].values
    labels = combined_df[label_column].values.reshape(-1, 1)

    return features, labels


def get_all_subjects(config):
    """
    Scans the data directory and returns a sorted list of all unique subject IDs.
    """
    data_dir = config.DATA_DIR
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    subjects = sorted(list(set([os.path.basename(f).split('_')[0] for f in all_files])))
    return subjects


def load_and_split_data(config, all_subjects, test_idx, val_idx):
    """
    Splits data files into train, validation, and test sets based on subject IDs.
    Uses test_idx and val_idx to determine the split. If they are None,
    performs a random split.

    Args:
        config: The project configuration object.
        all_subjects (list): Sorted list of all subject IDs.
        test_idx (int or None): The index of the subject to use for testing.
        val_idx (int or None): The index of the subject to use for validation.

    Returns:
        tuple: (train_files, val_files, test_files, train_subjects)
    """
    print("Loading and splitting data by subject...")
    data_dir = config.DATA_DIR
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    num_subjects = len(all_subjects)

    if num_subjects < 6:
        print(f"Warning: Only {num_subjects} subjects found. The 4:1:1 split may not be ideal.")

    # CV Mode: Split is determined by passed-in indices
    if test_idx is not None and val_idx is not None:
        print(f"--- Using CV Fold: Test Idx={test_idx}, Val Idx={val_idx} ---")
        if test_idx == val_idx:
            raise ValueError(f"Test index ({test_idx}) and Val index ({val_idx}) cannot be the same.")
        if not (0 <= test_idx < num_subjects and 0 <= val_idx < num_subjects):
            raise ValueError(f"Indices ({test_idx}, {val_idx}) are out of bounds for {num_subjects} subjects.")

        test_subjects = [all_subjects[test_idx]]
        val_subjects = [all_subjects[val_idx]]
        train_subjects = [s for i, s in enumerate(all_subjects) if i != test_idx and i != val_idx]

    # Random Mode: Split is determined by random shuffle
    else:
        print(f"--- Using Random Shuffle Mode (Seed: {config.RANDOM_SEED}) ---")
        random.seed(config.RANDOM_SEED)
        shuffled_subjects = all_subjects.copy()
        random.shuffle(shuffled_subjects)

        # Split subjects (e.g., 4 train, 1 val, 1 test)
        train_subjects = shuffled_subjects[:4]
        val_subjects = shuffled_subjects[4:5]
        test_subjects = shuffled_subjects[5:6]

    print(f"Training subjects: {train_subjects}")
    print(f"Validation subjects: {val_subjects}")
    print(f"Test subjects: {test_subjects}")

    # Assign files to sets based on subject ID
    train_files = [f for f in all_files if os.path.basename(f).split('_')[0] in train_subjects]
    val_files = [f for f in all_files if os.path.basename(f).split('_')[0] in val_subjects]
    test_files = [f for f in all_files if os.path.basename(f).split('_')[0] in test_subjects]

    return train_files, val_files, test_files, train_subjects


# --- 3. Scaler Handling ---

def get_or_create_scaler(config, scaler_path, all_subjects, test_idx, val_idx):
    """
    Loads a scaler from scaler_path if available.
    If not, it creates, fits (using the correct training split), and saves a new Scaler.

    Args:
        config: The project configuration object.
        scaler_path (str): The specific path to load/save the scaler for this fold.
        all_subjects (list): Sorted list of all subject IDs.
        test_idx (int or None): The test subject index (for split).
        val_idx (int or None): The validation subject index (for split).

    Returns:
        sklearn.preprocessing.StandardScaler: The fitted scaler.
    """
    if os.path.exists(scaler_path):
        print(f"Loading existing Scaler from '{scaler_path}'.")
        scaler = joblib.load(scaler_path)
        return scaler

    print(f"Scaler not found at '{scaler_path}'. Creating a new Scaler from training data...")

    # Load only training data files to fit the scaler
    # This now respects the CV or Random split
    train_files, _, _, _ = load_and_split_data(config, all_subjects, test_idx, val_idx)
    train_features, _ = load_data_from_files(train_files)

    if train_features.size == 0:
        raise ValueError("Could not load training data to create Scaler.")

    scaler = StandardScaler()
    scaler.fit(train_features)

    # Save the fitted scaler for this specific fold
    joblib.dump(scaler, scaler_path)
    print(f"New Scaler created and saved to '{scaler_path}'.")
    return scaler


# --- 4. DataLoader Creation ---

def get_data_loaders(config, scaler, all_subjects, test_idx, val_idx):
    """
    Creates and returns train, validation, and test DataLoaders for a specific split.

    Args:
        config: The project configuration object.
        scaler: The fitted StandardScaler.
        all_subjects (list): Sorted list of all subject IDs.
        test_idx (int or None): The test subject index (for split).
        val_idx (int or None): The validation subject index (for split).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # This call now respects the CV or Random split
    train_files, val_files, test_files, _ = load_and_split_data(config, all_subjects, test_idx, val_idx)

    print("\nLoading data from files...")
    train_features, train_labels = load_data_from_files(train_files)
    val_features, val_labels = load_data_from_files(val_files)
    test_features, test_labels = load_data_from_files(test_files)

    if train_features.size == 0 or val_features.size == 0 or test_features.size == 0:
        raise ValueError("Error: At least one data split (train, val, test) is empty.")

    print("\nApplying StandardScaler transformation...")
    # Apply the *same* fitted scaler to all datasets
    train_features_scaled = scaler.transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)
    print("Standardization complete.")

    # Create Dataset instances
    train_dataset = PPGDataset(train_features_scaled, train_labels, config.SEQUENCE_LENGTH)
    val_dataset = PPGDataset(val_features_scaled, val_labels, config.SEQUENCE_LENGTH)
    test_dataset = PPGDataset(test_features_scaled, test_labels, config.SEQUENCE_LENGTH)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader