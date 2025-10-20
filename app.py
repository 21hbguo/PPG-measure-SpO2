# app.py
import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import torch
import joblib
import tempfile
import time
from torch.utils.data import DataLoader
from collections import deque
import plotly.graph_objects as go

# Import modules from your project
import config
from models import get_model_class
# PredictionDataset is still needed for the generator
from data_utils import PredictionDataset
from utils import load_checkpoint


# --- 1. Helper Functions (Core Logic) ---

@st.cache_data
def find_checkpoints(results_dir):
    """
    Scans the results directory to find all best.pth files.
    """
    checkpoints = []
    for root, dirs, files in os.walk(results_dir):
        if "best.pth" in files:
            checkpoints.append(os.path.join(root, "best.pth"))
    relative_paths = [os.path.relpath(p, results_dir) for p in checkpoints]
    checkpoint_map = {os.path.join(results_dir, p): p for p in relative_paths}
    return checkpoint_map


@st.cache_resource
def load_model_and_scaler(checkpoint_path):
    """
    Intelligently loads the model and its corresponding scaler based on the checkpoint path.
    """
    st.write(f"Loading model: {checkpoint_path}")
    model_dir = os.path.dirname(checkpoint_path)
    model_name_dir = os.path.dirname(model_dir)

    if os.path.dirname(model_name_dir) == config.RESULTS_DIR:
        model_name = os.path.basename(model_name_dir)
    elif model_name_dir == config.RESULTS_DIR:
        model_name = os.path.basename(model_dir)
    else:
        st.error(f"Could not determine model name from path: {checkpoint_path}")
        return None, None, None

    fold_name = os.path.basename(model_dir)

    if "fold_test" in fold_name:
        scaler_name = f"scaler_{fold_name.split('_', 1)[1]}.pkl"
        scaler_path = os.path.join(config.RESULTS_DIR, scaler_name)
    else:
        scaler_path = config.DEFAULT_SCALER_PATH

    if not os.path.exists(scaler_path):
        st.error(f"Error: Corresponding scaler not found!\nExpected at: {scaler_path}")
        return None, None, None

    st.write(f"Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)

    st.write(f"Looking for model config: '{model_name}'")
    try:
        model_args = config.MODEL_CONFIGS[model_name]
    except KeyError:
        st.error(f"Error: Config for model '{model_name}' not found in config.py.")
        return None, None, None

    ModelClass = get_model_class(model_name)
    model = ModelClass(**model_args)
    model = load_checkpoint(model, checkpoint_path, config.DEVICE)
    model.eval()

    return model, scaler, model_name


# --- Optimized Generator with Batch Prediction ---
def process_and_predict_generator(uploaded_file, model, scaler, batch_size, progress_bar=None, target_fps=30):
    """
    Processes a video by first extracting all frames, then running inference in batches.
    Finally, yields results one by one for playback.
    """
    # --- Stage 1: Fast Frame Extraction ---
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        os.remove(temp_video_path)
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = int(original_fps / target_fps) if original_fps > target_fps else 0

    st.write(f"Video Info: {frame_total} frames, {original_fps:.2f} FPS.")
    st.write(f"Target FPS: {target_fps} FPS (Processing 1 frame every {frame_skip + 1} frames).")

    all_raw_frames = []
    all_frame_indices = []

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (frame_skip + 1) == 0:
                b, g, r = cv2.split(frame)
                r_mean = np.mean(r.astype(np.float64) * 1)
                g_mean = np.mean(g.astype(np.float64) * 3)
                b_mean = np.mean(b.astype(np.float64) * 18)
                all_raw_frames.append([r_mean, g_mean, b_mean])
                all_frame_indices.append(frame_count)

            frame_count += 1
            if progress_bar:
                progress_bar.progress(frame_count / frame_total,
                                      text=f"Extracting video frames: {frame_count}/{frame_total}")
    finally:
        cap.release()
        os.remove(temp_video_path)

    if not all_raw_frames:
        return

    # --- Stage 2: High-Speed Batch Prediction ---
    with st.spinner("Running batch prediction on all frames..."):
        features = np.array(all_raw_frames)
        features_scaled = scaler.transform(features)

        pred_dataset = PredictionDataset(features_scaled, config.SEQUENCE_LENGTH)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        with torch.no_grad():
            for seq_batch in pred_loader:
                seq_batch = seq_batch.to(config.DEVICE)
                outputs = model(seq_batch)
                all_predictions.extend(outputs.cpu().numpy().flatten())

    # --- Stage 3: Yield Results for Playback ---
    center_offset_frames = (config.SEQUENCE_LENGTH // 2)
    num_predictions = len(all_predictions)

    for i in range(num_predictions):
        original_frame_index = all_frame_indices[i + center_offset_frames]
        current_time_sec = original_frame_index / original_fps

        yield {
            "time_sec": current_time_sec,
            "predicted_SpO2": all_predictions[i],
            "R": features[i + center_offset_frames, 0],
            "G": features[i + center_offset_frames, 1],
            "B": features[i + center_offset_frames, 2],
        }


# --- 2. Optimized UI & Playback ---

@st.cache_data
def create_spo2_gauge(value, threshold, min_val=80, max_val=100):
    """
    Creates a Plotly Gauge chart.
    """
    value = max(min_val, min(max_val, value))  # Clamp value

    # Set colors based on threshold
    if value >= threshold:
        bar_color = "#00FF00"  # Green
        threshold_color = "gray"
    else:
        bar_color = "#FF0000"  # Red
        threshold_color = "#FF0000"  # Red threshold line for alert

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%", 'font': {'size': 50}, 'valueformat': '.1f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>Current SpO2</b><br>(Threshold: {threshold}%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, threshold], 'color': 'rgba(255, 0, 0, 0.1)'},
                {'range': [threshold, max_val], 'color': 'rgba(0, 255, 0, 0.1)'}
            ],
            'threshold': {
                'line': {'color': threshold_color, 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def display_results_playback(results_df, placeholders, alert_threshold, display_window, playback_speed):
    """
    Displays the smooth playback using Plotly Gauge and Line Chart.
    """
    gauge_placeholder, rgb_chart_placeholder = placeholders
    plot_buffer = deque()
    sleep_duration = 0 if playback_speed == float('inf') else 1.0 / playback_speed

    # Set gauge display range
    min_spo2_display = 80
    max_spo2_display = 100

    for index, result in results_df.iterrows():
        plot_buffer.append(result.to_dict())

        # Maintain deque window size
        while len(plot_buffer) > 1 and (plot_buffer[-1]['time_sec'] - plot_buffer[0]['time_sec']) > display_window:
            plot_buffer.popleft()

        df_window = pd.DataFrame(list(plot_buffer))
        current_spo2 = result['predicted_SpO2']

        # 1. Update smooth gauge
        fig_gauge = create_spo2_gauge(current_spo2, alert_threshold, min_spo2_display, max_spo2_display)
        gauge_placeholder.plotly_chart(fig_gauge, use_container_width=True)

        # 2. Update scrolling RGB chart
        rgb_chart_placeholder.line_chart(
            df_window.rename(columns={'time_sec': 'second'}).set_index('second')[['R', 'G', 'B']])

        if sleep_duration > 0:
            time.sleep(sleep_duration)

    # Clear placeholders
    gauge_placeholder.empty()
    rgb_chart_placeholder.empty()


def main():
    st.set_page_config(page_title="PPG SpO2 Analyzer", layout="wide")
    st.title("📹 PPG Signal SpO2 Prediction Analyzer")

    if 'prediction_results' not in st.session_state:
        st.session_state['prediction_results'] = None
    if 'last_video_name' not in st.session_state:
        st.session_state['last_video_name'] = None

    with st.sidebar:
        st.header("1. Upload Video")
        video_file = st.file_uploader("Upload a .mp4 video file", type=["mp4"])

        st.header("2. Model Selection")
        checkpoint_map = find_checkpoints(config.RESULTS_DIR)
        if not checkpoint_map:
            st.error(f"No 'best.pth' models found in '{config.RESULTS_DIR}'. Please train a model first.")
            return

        selected_checkpoint_name = st.selectbox("Select a trained model:", options=list(checkpoint_map.values()))
        selected_checkpoint_path = [k for k, v in checkpoint_map.items() if v == selected_checkpoint_name][0]

        st.header("3. Analysis Settings")
        alert_threshold = st.slider("Low SpO2 Alert Threshold (%)", 85, 99, 92,
                                    help="The gauge will turn red if the SpO2 prediction falls below this value.")
        display_window = st.number_input("Scrolling Display Window (seconds)", min_value=10, max_value=300, value=30,
                                         step=5,
                                         help="How many recent seconds of data to show on the scrolling RGB chart during playback.")
        batch_size = st.number_input("Prediction Batch Size", min_value=1, max_value=4096, value=256, step=64,
                                     help="Increase this value for faster GPU prediction, but it will consume more VRAM.")
        speed_options = {"1x": 1, "2x": 2, "5x": 5, "10x": 10, "Max Speed": float('inf')}
        selected_speed_str = st.selectbox("Select playback speed:", options=list(speed_options.keys()))
        playback_speed = speed_options[selected_speed_str]

        st.header("4. Controls")
        run_button_text = "Start Playback" if st.session_state.prediction_results is not None else "Process Video & Analyze"
        run_button = st.button(run_button_text, disabled=(video_file is None))

        st.divider()
        st.info(f"Using device for inference: **{str(config.DEVICE).upper()}**")

    # --- State Management: If video changes, clear old results ---
    if video_file and video_file.name != st.session_state.last_video_name:
        st.session_state.prediction_results = None
        st.session_state.last_video_name = video_file.name
        st.rerun()

    # --- Core Processing Logic ---
    if run_button and video_file is not None:
        # 1. Only process video if no results exist
        if st.session_state.prediction_results is None:
            with st.spinner("Loading model and scaler..."):
                model, scaler, model_name = load_model_and_scaler(selected_checkpoint_path)
                if model is None: st.stop()
                st.success(f"Successfully loaded model '{model_name}' and its scaler.")

            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0, text="Processing video...")

            # Use generator to process video
            live_generator = process_and_predict_generator(video_file, model, scaler, batch_size, progress_bar)
            all_results = list(live_generator)

            progress_placeholder.empty()

            if all_results:
                st.session_state.prediction_results = pd.DataFrame(all_results)
                st.success("Video processing and model prediction complete!")
            else:
                st.warning("No results were generated after processing the video.")
                st.stop()

        # 2. If results exist in session_state, display tabs
        if st.session_state.prediction_results is not None:
            tab_playback, tab_report = st.tabs(["📈 Dynamic Playback", "📊 Full Analysis Report"])

            # --- Tab 1: Dynamic Playback ---
            with tab_playback:
                st.header(f"Dynamic Playback: {selected_checkpoint_name}")

                # Beautification: Wrap charts in a styled container
                with st.container(border=True):
                    col1, col2 = st.columns([0.6, 0.4])
                    with col1:
                        gauge_placeholder = st.empty()
                    with col2:
                        rgb_chart_placeholder = st.empty()

                    # Initialize empty placeholders
                    gauge_placeholder.plotly_chart(create_spo2_gauge(98.0, alert_threshold),
                                                   use_container_width=True)  # Show a default gauge
                    rgb_chart_placeholder.line_chart(pd.DataFrame(columns=['R', 'G', 'B']))  # Show an empty chart

                placeholders = (gauge_placeholder, rgb_chart_placeholder)

                display_results_playback(
                    st.session_state.prediction_results,
                    placeholders,
                    alert_threshold,
                    display_window,
                    playback_speed
                )
                st.info("Playback finished.")

            # --- Tab 2: Full Analysis Report ---
            with tab_report:
                st.header("Full Analysis Report")
                full_df = st.session_state.prediction_results

                # Calculate 'Normal' and 'Alert' for the report
                full_df['Normal'] = np.where(full_df['predicted_SpO2'] >= alert_threshold, full_df['predicted_SpO2'],
                                             np.nan)
                full_df['Alert'] = np.where(full_df['predicted_SpO2'] < alert_threshold, full_df['predicted_SpO2'],
                                            np.nan)

                # Beautification: Wrap metrics in a styled container
                with st.container(border=True):
                    st.subheader("Statistical Summary")
                    total_duration = full_df['time_sec'].max() - full_df['time_sec'].min()
                    alert_count = full_df['Alert'].notna().sum()
                    alert_percentage = (alert_count / len(full_df)) * 100 if len(full_df) > 0 else 0

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Average SpO2", f"{full_df['predicted_SpO2'].mean():.1f}%")
                    col2.metric("Minimum SpO2", f"{full_df['predicted_SpO2'].min():.1f}%")
                    col3.metric("Maximum SpO2", f"{full_df['predicted_SpO2'].max():.1f}%")
                    col4.metric("Low SpO2 Event %", f"{alert_percentage:.1f}%")

                # Beautification: Wrap charts in a styled container
                with st.container(border=True):
                    st.subheader("Full Data Charts")
                    col1_chart, col2_chart = st.columns(2)
                    with col1_chart:
                        st.line_chart(
                            full_df.rename(columns={'time_sec': 'second'}).set_index('second')[['Normal', 'Alert']],
                            color=["#00FF00", "#FF0000"])
                    with col2_chart:
                        st.line_chart(
                            full_df.rename(columns={'time_sec': 'second'}).set_index('second')[['R', 'G', 'B']])

                @st.cache_data
                def convert_df_to_csv(df):
                    df_to_download = df[['time_sec', 'predicted_SpO2', 'R', 'G', 'B']]
                    return df_to_download.to_csv(index=False).encode('utf-8')

                csv_data = convert_df_to_csv(full_df)
                st.download_button(
                    label="Download results as CSV",
                    data=csv_data,
                    file_name=f"prediction_results_{st.session_state.last_video_name.split('.')[0]}.csv",
                    mime='text/csv',
                )

    # If results already exist but button wasn't pressed, still show report
    elif st.session_state.prediction_results is not None:
        st.info("Loaded previous results. Click 'Start Playback' or view the 'Full Analysis Report' tab.")

        tab_playback, tab_report = st.tabs(["📈 Dynamic Playback", "📊 Full Analysis Report"])

        with tab_playback:
            st.info("Click the 'Start Playback' button in the sidebar to begin the dynamic demo.")

        with tab_report:
            st.header("Full Analysis Report")
            full_df = st.session_state.prediction_results

            # Check if columns exist, prevent recalculation
            if 'Normal' not in full_df.columns:
                full_df['Normal'] = np.where(full_df['predicted_SpO2'] >= alert_threshold, full_df['predicted_SpO2'],
                                             np.nan)
                full_df['Alert'] = np.where(full_df['predicted_SpO2'] < alert_threshold, full_df['predicted_SpO2'],
                                            np.nan)
                st.session_state.prediction_results = full_df

            # Beautification: Wrap metrics in a styled container
            with st.container(border=True):
                st.subheader("Statistical Summary")
                alert_count = full_df['Alert'].notna().sum()
                alert_percentage = (alert_count / len(full_df)) * 100 if len(full_df) > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average SpO2", f"{full_df['predicted_SpO2'].mean():.1f}%")
                col2.metric("Minimum SpO2", f"{full_df['predicted_SpO2'].min():.1f}%")
                col3.metric("Maximum SpO2", f"{full_df['predicted_SpO2'].max():.1f}%")
                col4.metric("Low SpO2 Event %", f"{alert_percentage:.1f}%")

            # Beautification: Wrap charts in a styled container
            with st.container(border=True):
                st.subheader("Full Data Charts")
                col1_chart, col2_chart = st.columns(2)
                with col1_chart:
                    st.line_chart(
                        full_df.rename(columns={'time_sec': 'second'}).set_index('second')[['Normal', 'Alert']],
                        color=["#00FF00", "#FF0000"])
                with col2_chart:
                    st.line_chart(full_df.rename(columns={'time_sec': 'second'}).set_index('second')[['R', 'G', 'B']])

            @st.cache_data
            def convert_df_to_csv_cached(df):
                df_to_download = df[['time_sec', 'predicted_SpO2', 'R', 'G', 'B']]
                return df_to_download.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv_cached(full_df)
            st.download_button(
                label="Download results as CSV",
                data=csv_data,
                file_name=f"prediction_results_{st.session_state.last_video_name.split('.')[0]}.csv",
                mime='text/csv',
            )

if __name__ == '__main__':
    main()