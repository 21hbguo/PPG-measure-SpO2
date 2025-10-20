# preprocess_data.py
"""
Data Preprocessing Script.

This script processes raw MP4 videos and ground truth (GT) CSV files to generate
model-ready, time-aligned datasets.

Workflow:
1. Iterates through all MP4 files in the specified video directory.
2. For each video, it locates the corresponding GT file based on a shared ID.
3. It loads the GT data, calculates an average SpO2, and determines the
   start timestamp.
4. It reads the video frame by frame, extracting the mean R, G, B values.
5. Applies channel-specific gains as described in the source research.
6. Generates a timestamp for each frame, synchronized with the GT start time.
7. Merges the RGB data with the corresponding GT SpO2 value based on the timestamp.
8. Cleans the final dataset by removing rows with missing or invalid SpO2 values.
9. Saves the processed data to a new CSV file in the output directory.
"""

import os
import cv2
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
# <-- 从 config.py 导入路径
from config import VIDEO_ROOT_DIR, GT_DATA_DIR, DATA_DIR as OUTPUT_DIR


def load_ground_truth(gt_path):
    """
    Loads and preprocesses a ground truth (GT) CSV file.

    Args:
        gt_path (str): Path to the ground truth CSV file.

    Returns:
        tuple: A tuple containing:
            - spo2_lookup (dict): A dictionary mapping timestamps (str) to SpO2 values (float).
            - start_seconds (int): The total seconds from midnight for the first timestamp.
    """
    if not os.path.exists(gt_path):
        print(f"Warning: GT file not found at {gt_path}. Timestamps will start from 00:00:00.")
        return {}, 0

    try:
        gt_df = pd.read_csv(gt_path)

        # As per the paper, specific Masimo device data is preferred.
        # This is a simplified approach using an average of available SpO2 columns.
        spo2_columns = ['SpO2 1', 'SpO2 2', 'SpO2 4', 'SpO2 5']
        gt_df['average_SpO2'] = gt_df[spo2_columns].mean(axis=1)

        spo2_lookup = gt_df.set_index('Time')['average_SpO2'].to_dict()

        # Determine the start time in total seconds from the first entry
        start_time_str = gt_df['Time'].iloc[0]
        h, m, s = map(int, start_time_str.split(':'))
        start_seconds = h * 3600 + m * 60 + s

        print(f"Successfully loaded GT data. Start time aligned to {start_time_str}.")
        return spo2_lookup, start_seconds

    except Exception as e:
        print(f"Warning: Failed to process GT file {gt_path}: {e}. Timestamps will start from 0.")
        return {}, 0


def process_video_file(video_path, spo2_lookup, start_seconds, output_path):
    """
    Processes a single video file, extracts RGB data, and merges it with GT data.

    Args:
        video_path (str): Path to the MP4 video file.
        spo2_lookup (dict): Timestamp-to-SpO2 mapping from the GT file.
        start_seconds (int): The starting timestamp in total seconds.
        output_path (str): Path to save the processed output CSV file.
    """
    # Video properties based on the research paper
    FPS = 30.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_index', 'timestamp', 'R', 'G', 'B', 'average_SpO2'])

        for frame_count in tqdm(range(frame_total), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the timestamp for the current frame
            elapsed_seconds = frame_count / FPS
            current_total_seconds = start_seconds + elapsed_seconds

            hours = int(current_total_seconds // 3600)
            minutes = int((current_total_seconds % 3600) // 60)
            seconds = int(current_total_seconds % 60)
            timestamp = f" {hours:02d}:{minutes:02d}:{seconds:02d}"

            # Split channels and calculate the mean pixel value
            b, g, r = cv2.split(frame)

            # Apply channel gains as described in the paper
            r_mean = round(np.mean(r.astype(np.float64) * 1), 6)  # 1x gain
            g_mean = round(np.mean(g.astype(np.float64) * 3), 6)  # 3x gain
            b_mean = round(np.mean(b.astype(np.float64) * 18), 6)  # 18x gain

            # Look up the SpO2 value for the current timestamp
            avg_spo2_value = spo2_lookup.get(timestamp, "")

            csv_writer.writerow([frame_count, timestamp, r_mean, g_mean, b_mean, avg_spo2_value])

    cap.release()


def clean_processed_file(file_path):
    """
    Cleans a processed CSV file by removing rows with invalid SpO2 values.

    Args:
        file_path (str): Path to the processed CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        rows_before = len(df)

        # Replace empty strings with NaN to facilitate dropping
        df['average_SpO2'].replace('', np.nan, inplace=True)
        df.dropna(subset=['average_SpO2'], inplace=True)

        # Filter out values below the plausible threshold of 70%
        df = df[df['average_SpO2'] >= 70]
        rows_after = len(df)

        df.to_csv(file_path, index=False)
        print(f"Cleaned {os.path.basename(file_path)}: Removed {rows_before - rows_after} rows.")
    except Exception as e:
        print(f"Error while cleaning file {file_path}: {e}")


def main():
    """Main function to run the batch processing workflow."""

    # --- Configuration ---
    # <-- 路径现在从 config.py 加载
    # 我们使用 DATA_DIR (从 config 导入并重命名为 OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting data preprocessing.")
    print(f"Input Videos: {VIDEO_ROOT_DIR}")
    print(f"Input GT: {GT_DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    # --- Batch Processing ---
    for dirpath, _, filenames in os.walk(VIDEO_ROOT_DIR):
        for filename in filenames:
            print(filename, '------------------')
            if filename.lower().endswith('.mp4'):
                video_full_path = os.path.join(dirpath, filename)

                # Extract subject ID (e.g., "100001") from filename
                subject_id = filename.split('-')[0]

                # Determine output filename suffix (_Left, _Right)
                suffix = ""
                if "Left" in dirpath:
                    suffix = "_Left"
                elif "Right" in dirpath:
                    suffix = "_Right"

                output_csv_name = f"{subject_id}{suffix}.csv"
                output_csv_path = os.path.join(OUTPUT_DIR, output_csv_name)

                print(f"\nProcessing: {filename} -> {output_csv_name}")

                # Load corresponding ground truth data
                gt_csv_path = os.path.join(GT_DATA_DIR, f"{subject_id}.csv")
                spo2_lookup, start_seconds = load_ground_truth(gt_csv_path)

                # Process the video and save initial CSV
                process_video_file(video_full_path, spo2_lookup, start_seconds, output_csv_path)

                # Clean the resulting CSV file
                clean_processed_file(output_csv_path)

    print("\nBatch processing complete.")


if __name__ == '__main__':
    main()