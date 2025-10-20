# PPG-based SpO2 Prediction Framework
This project offers a **modular, extensible framework** for estimating blood oxygen saturation (SpO2). It processes raw video feeds to extract Photoplethysmography (PPG) signals, then uses these signals to predict SpO2. The codebase includes dedicated scripts for data preprocessing, model training, and inference, with a structure that simplifies experimentation with different deep learning architectures.


## Key Features
- **End-to-End Workflow**: Covers the full pipeline from raw video processing, through model training, to final SpO2 prediction.
- **Model Agnostic**: Uses a "Model Factory" design. Switch or test different architectures (e.g., CNN, LSTM) by only modifying one line in the configuration file.
- **Reproducible Science**: All critical parameters—hyperparameters, file paths, and random seeds—are centralized in `config.py` to ensure consistent results.
- **Organized Outputs**: Training artifacts (model weights, evaluation plots) are auto-saved to clearly named subdirectories (e.g., `results/CNN/`), one per model.
- **Easy Extension**: Clean, well-commented, and modular code. Add new models, data processing steps, or evaluation metrics with minimal changes.


## Project Workflow
The framework follows three core stages, each with clear input, processing logic, and output.

### 1. Preprocessing (`preprocess_data.py`)
- **Input**: Raw video files (.mp4) and ground truth data (.csv).
- **Process**: Extracts RGB signals from video frames, applies channel gains, synchronizes signals with ground truth timestamps, and cleans the data.
- **Output**: Processed .csv files ready for model training.

### 2. Training (`train.py`)
- **Input**: Processed .csv files from the preprocessing stage.
- **Process**: Loads data, creates a global `StandardScaler` for normalization, trains the model specified in `config.py`, and evaluates model performance.
- **Output**: A shared `scaler.pkl` (for data normalization) and model-specific artifacts (e.g., `results/CNN/best.pth` weights file, training loss plots).

### 3. Prediction (`predict.py`)
- **Input**: A processed .csv file (for inference) and a trained model.
- **Process**: Loads the pre-saved `scaler.pkl` and model weights, then runs inference on the input data.
- **Output**: A visualization plot comparing predicted SpO2 values with ground truth.


## Project Structure
```
spo2_project/
├── 📂 data/                  # Data storage (input + processed)
│   ├── gt/                   # Input: Ground truth CSV files
│   ├── raw_videos/           # Input: Raw MP4 video files
│   └── processed/            # Output: Preprocessed CSV (input for training)
├── 📂 models/                # Model definitions & factory
│   ├── __init__.py           # Model registry (links model names to classes)
│   ├── cnn_model.py          # CNN architecture
│   └── lstm_model.py         # LSTM architecture (example)
├── 📂 results/               # Training outputs
│   ├── scaler.pkl            # Global data scaler (shared across models)
│   └── 📂 CNN/               # Model-specific artifacts (example)
│       ├── best.pth          # Best-trained weights for CNN
│       └── ...               # Evaluation plots, logs, etc.
├── preprocess_data.py        # Run data preprocessing
├── config.py                 # Central config (hyperparameters, paths, model name)
├── data_utils.py             # PyTorch Datasets & DataLoaders (for training)
├── utils.py                  # Helper functions (plotting, random seed setup)
├── train.py                  # Run model training
├── predict.py                # Run inference/prediction
├── app.py                    # Web application
├── requirements.txt          # Python dependencies (e.g., PyTorch, OpenCV)
└── README.md                 # Project documentation (this file)
```


## How to Use
Follow these 5 steps to set up the framework, process data, train a model, and run predictions.

### Step 1: Set Up the Environment
First, clone the repo and install dependencies. Use a virtual environment to avoid package conflicts.
```bash
# Clone the repository
git clone <repository_url>
cd spo2_project

# Create and activate a virtual environment 
conda create --name spo2 python=3.13
conda activate spo2

# Install required packages
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
Place your raw data in the correct folders (the script auto-detects files, including in subdirectories):
- Raw videos (.mp4): Move to `data/raw_videos/` (e.g., `100001-right.mp4`).
- Ground truth CSVs: Move to `data/gt/` (e.g., `100001.csv`).

### Step 3: Run Data Preprocessing
Convert raw videos into model-ready time-series data with one command:
```bash
python preprocess_data.py
```
The script reads from `data/raw_videos/` and `data/gt/`, then saves processed CSVs to `data/processed/`.

### Step 4: Configure and Train a Model
1. Open `config.py` and verify/set the following:
   - Paths: Ensure `RAW_VIDEO_DIR`, `GT_DIR`, and `PROCESSED_DATA_DIR` point to the correct subfolders in `data/`.
   - Model: Set `MODEL_NAME` to your target model (e.g., `"CNN"` or `"LSTMModel"`).
   - (Optional) Hyperparameters: Adjust `EPOCHS`, `BATCH_SIZE`, learning rate, etc.
2. Run the training script:
```bash
python train.py
```
All outputs (weights, plots) are saved to `results/<MODEL_NAME>/` (e.g., `results/CNN/`).

### Step 5: Run Prediction
Use a trained model to predict SpO2 on new processed data:
1. Open `predict.py` and find the `if __name__ == '__main__':` block.
2. Set `CSV_TO_PREDICT` to the path of your processed CSV (e.g., `data/processed/100001_Left.csv`).
3. Confirm `MODEL_NAME` in `config.py` matches the trained model you want to use.
4. Run the prediction script:
```bash
python predict.py
```


## How to Add a New Model
The framework is designed for easy extension. To add a new model (e.g., Transformer), follow these 4 steps:

1. **Create a Model File**:  
   Add `models/transformer_model.py` and define your PyTorch model class (e.g., `class Transformer(torch.nn.Module): ...`).

2. **Register the Model**:  
   Open `models/__init__.py` and:
   - Import your new class: `from .transformer_model import Transformer`.
   - Add it to `MODEL_REGISTRY`: `MODEL_REGISTRY = {"CNN": CNN, "LSTM": LSTM, "Transformer": Transformer}`.
3. **Add Model Configuration**:  
   Open `config.py` and add a new entry to `MODEL_CONFIGS` (include all arguments your model needs to initialize):
   ```python
   MODEL_CONFIGS = {
       "CNN": {"input_dim": 3, "hidden_dim": 64},
       "LSTM": {"input_dim": 3, "hidden_dim": 128, "num_layers": 2},
       "Transformer": {"input_dim": 3, "d_model": 64, "nhead": 4}  # New model config
   }
   ```

4. **Train the New Model**:  
   In `config.py`, set `MODEL_NAME = "Transformer"`, then run `python train.py`. The framework will handle the rest.

## 🚀 Interactive Web Application (`app.py`)

In addition to the command-line scripts, this project includes a powerful, interactive web application built with Streamlit for real-time analysis and visualization.

This app provides a user-friendly interface to:

1.  **Upload** a raw video file (`.mp4`).
2.  **Select** any trained model found in the `results/` directory.
3.  **Run** on-the-fly processing and prediction.
4.  **Visualize** the results with a dynamic, real-time playback dashboard.
5.  **Review** a full statistical report and download the prediction data.

### App Features

  - **Interactive Video Upload**: Users can directly upload a `.mp4` video file through the web interface.
  - **Automatic Model Discovery**: The app automatically scans the `results/` directory and populates a dropdown menu to select any trained `best.pth` model.
  - **Optimized On-the-Fly Processing**: Performs high-speed frame extraction and batched GPU inference (using the same logic as the core framework) to generate predictions for the entire video.
  - **Dynamic Playback Tab**:
      - **SpO2 Gauge**: A real-time gauge (using Plotly) shows the current predicted SpO2, with a configurable alert threshold (e.g., turns red below 92%).
      - **Scrolling Signal Chart**: A live, scrolling line chart displays the raw R, G, and B signal values for a user-defined time window (e.g., 30 seconds).
      - **Playback Controls**: Adjust the playback speed (e.g., 1x, 5x, Max Speed).
  - **Full Analysis Report Tab**:
      - **Statistical Summary**: Displays key metrics like Average, Minimum, and Maximum SpO2, and the percentage of time spent in an "Alert" state.
      - **Complete Data Charts**: Static plots showing the *entire* SpO2 prediction and RGB signal time-series.
      - **Data Export**: A "Download results as CSV" button to save the predictions and timestamps.

### How to Run the Web App

Ensure you have installed all dependencies from `requirements.txt` (which must include `streamlit`).

1.  Make sure you have at least one trained model saved in the `results/` folder (e.g., `results/CNN/best.pth`).
2.  Run the following command from your project's root directory:

<!-- end list -->

```bash
streamlit run app.py
```

3.  The application will open automatically in your web browser.