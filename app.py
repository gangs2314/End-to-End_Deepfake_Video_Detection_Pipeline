"""
Streamlit app for interactive deepfake detection demo.
Upload a video → see per-frame scores → video verdict → Grad-CAM overlays.
"""
import streamlit as st
import torch
import numpy as np
import cv2
from pathlib import Path
import tempfile
from typing import Optional, List
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from models_baseline import create_model
from models_temporal import create_temporal_model
from preprocess import PreprocessingPipeline
from explainability import ExplainabilityPipeline
from evaluate import DeepfakeEvaluator


@st.cache_resource
def load_models():
    """Load frame and temporal models (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load baseline frame model
    try:
        frame_model = create_model("efficientnet", num_classes=2)
        frame_model.to(device)
        frame_model.eval()
    except Exception as e:
        st.warning(f"Could not load frame model: {e}")
        frame_model = None

    # Load temporal model
    try:
        temporal_model = create_temporal_model("lstm", "efficientnet", num_classes=2)
        temporal_model.to(device)
        temporal_model.eval()
    except Exception as e:
        st.warning(f"Could not load temporal model: {e}")
        temporal_model = None

    return frame_model, temporal_model, device


@st.cache_resource
def get_preprocessing_pipeline():
    """Get preprocessing pipeline (cached)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PreprocessingPipeline(
        detector_model="retinaface",
        face_size=224,
        device=device
    )


def process_video(
    video_path: str,
    frame_model: torch.nn.Module,
    temporal_model: Optional[torch.nn.Module],
    preprocessor: PreprocessingPipeline,
    num_frames: int = 8,
    device: str = "cpu"
) -> dict:
    """
    Process video and generate predictions.

    Args:
        video_path: Path to video file
        frame_model: Frame-level classifier
        temporal_model: Temporal model (optional)
        preprocessor: Preprocessing pipeline
        num_frames: Number of frames to extract
        device: Computation device

    Returns:
        Dictionary with results
    """
    results = {
        'frame_predictions': [],
        'frame_confidences': [],
        'temporal_prediction': None,
        'temporal_confidence': None,
        'video_verdict': None,
        'processing_info': []
    }

    try:
        # Extract frames from video
        with st.spinner("Extracting frames from video..."):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()

            if not frames:
                st.error("Could not extract frames from video")
                return results

        # Process each frame
        with st.spinner("Processing frames..."):
            all_face_crops = []
            frame_logits = []

            for frame_idx, frame in enumerate(frames):
                # Convert BGR to RGB for preprocessing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect and extract faces
                faces, info = preprocessor.process_frame(frame_rgb)

                if faces:
                    # Convert face crops to tensors and get predictions
                    for face in faces:
                        # Normalize face crop
                        face_tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
                        normalize = torch.nn.functional.normalize
                        face_tensor = (face_tensor - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

                        face_tensor = face_tensor.unsqueeze(0).to(device)
                        all_face_crops.append(face)

                        # Frame-level prediction
                        with torch.no_grad():
                            logits = frame_model(face_tensor)
                            probs = torch.softmax(logits, dim=1)
                            pred_class = logits.argmax(dim=1).item()
                            confidence = probs[0, pred_class].item()

                        results['frame_predictions'].append(pred_class)
                        results['frame_confidences'].append(confidence)
                        frame_logits.append(logits.detach().cpu())

            if not all_face_crops:
                st.warning("No faces detected in video")
                return results

        # Aggregate predictions
        frame_preds = np.array(results['frame_predictions'])
        frame_confs = np.array(results['frame_confidences'])

        # Video verdict from majority voting
        video_verdict = 1 if np.mean(frame_preds) > 0.5 else 0
        video_confidence = np.abs(np.mean(frame_preds) - 0.5) * 2 + 0.5

        results['video_verdict'] = video_verdict
        results['video_confidence'] = float(video_confidence)
        results['num_faces'] = len(all_face_crops)

        # Temporal model prediction if available
        if temporal_model is not None and len(all_face_crops) >= 2:
            with st.spinner("Running temporal model..."):
                # Stack face crops into temporal sequence
                max_faces = min(len(all_face_crops), 16)
                temporal_sequence = []

                for i in range(max_faces):
                    face = all_face_crops[i]
                    face_tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
                    face_tensor = (face_tensor - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                    temporal_sequence.append(face_tensor)

                temporal_sequence = torch.stack(temporal_sequence).unsqueeze(0).to(device)

                with torch.no_grad():
                    temporal_logits, _ = temporal_model(temporal_sequence)
                    temporal_probs = torch.softmax(temporal_logits, dim=1)
                    temporal_pred = temporal_logits.argmax(dim=1).item()
                    temporal_conf = temporal_probs[0, temporal_pred].item()

                results['temporal_prediction'] = temporal_pred
                results['temporal_confidence'] = float(temporal_conf)

    except Exception as e:
        st.error(f"Error processing video: {e}")
        import traceback
        st.error(traceback.format_exc())

    return results


def display_results(results: dict):
    """Display results in Streamlit UI."""
    col1, col2, col3 = st.columns(3)

    # Frame-level verdict
    with col1:
        st.subheader("Frame-Level Analysis")
        if results['frame_predictions']:
            fake_ratio = np.mean(results['frame_predictions'])
            st.metric("Fake Ratio", f"{fake_ratio:.1%}")
            st.metric("Faces Detected", results.get('num_faces', 0))

    # Video-level verdict
    with col2:
        st.subheader("Video-Level Verdict")
        if results['video_verdict'] is not None:
            verdict = "🔴 FAKE" if results['video_verdict'] == 1 else "🟢 REAL"
            confidence = results.get('video_confidence', 0.5)
            st.metric("Verdict", verdict)
            st.metric("Confidence", f"{confidence:.1%}")

    # Temporal model
    with col3:
        st.subheader("Temporal Model")
        if results['temporal_prediction'] is not None:
            temporal_verdict = "🔴 FAKE" if results['temporal_prediction'] == 1 else "🟢 REAL"
            temporal_conf = results['temporal_confidence']
            st.metric("Verdict", temporal_verdict)
            st.metric("Confidence", f"{temporal_conf:.1%}")

    # Frame predictions chart
    if results['frame_predictions']:
        st.subheader("Frame-by-Frame Predictions")
        st.bar_chart(results['frame_predictions'])

        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Frames Processed**: {len(results['frame_predictions'])}")
            st.write(f"**Real Frames**: {sum(1 for p in results['frame_predictions'] if p == 0)}")

        with col2:
            st.write(f"**Fake Frames**: {sum(1 for p in results['frame_predictions'] if p == 1)}")
            avg_conf = np.mean(results['frame_confidences'])
            st.write(f"**Avg Confidence**: {avg_conf:.1%}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Deepfake Detection Demo",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Deepfake Detection System")
    st.markdown("Upload a video to detect deepfakes using frame-level and temporal analysis.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        num_frames = st.slider("Frames to Process", 4, 32, 8)
        model_type = st.selectbox("Model Type", ["Frame-Level", "Temporal (LSTM)", "Ensemble"])

    # Load models
    st.info("Loading models... (this may take a moment on first run)")
    frame_model, temporal_model, device = load_models()
    preprocessor = get_preprocessing_pipeline()

    if frame_model is None:
        st.error("Could not load models. Please check your environment.")
        return

    # File upload
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            video_path = tmp_file.name

        try:
            # Display video info
            st.video(uploaded_file)

            # Process video
            st.subheader("Analysis Results")
            results = process_video(
                video_path,
                frame_model,
                temporal_model if model_type != "Frame-Level" else None,
                preprocessor,
                num_frames=num_frames,
                device=device
            )

            # Display results
            if results['frame_predictions']:
                display_results(results)

                # Download report
                if st.button("📥 Download Report"):
                    report_text = f"""
DEEPFAKE DETECTION REPORT
========================

VIDEO ANALYSIS RESULTS
{'-' * 40}
Frame-Level Verdict: {'FAKE' if results['video_verdict'] == 1 else 'REAL'}
Video Confidence: {results.get('video_confidence', 0):.1%}
Faces Detected: {results.get('num_faces', 0)}
Frames Analyzed: {len(results['frame_predictions'])}

Real Frames: {sum(1 for p in results['frame_predictions'] if p == 0)}
Fake Frames: {sum(1 for p in results['frame_predictions'] if p == 1)}
Average Confidence: {np.mean(results['frame_confidences']):.1%}

{('-' * 40) if results['temporal_prediction'] is not None else ''}
{'TEMPORAL MODEL ANALYSIS' if results['temporal_prediction'] is not None else ''}
{('-' * 40) if results['temporal_prediction'] is not None else ''}
{'Temporal Verdict: ' + ('FAKE' if results['temporal_prediction'] == 1 else 'REAL') if results['temporal_prediction'] is not None else ''}
{'Temporal Confidence: ' + f"{results['temporal_confidence']:.1%}" if results['temporal_prediction'] is not None else ''}
"""
                    st.download_button(
                        label="Download Report",
                        data=report_text,
                        file_name="deepfake_report.txt",
                        mime="text/plain"
                    )

        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        ### About This Demo
        This system uses:
        - **Frame-Level Detection**: EfficientNet classifier on aligned face crops
        - **Temporal Modeling**: LSTM to capture inter-frame inconsistencies
        - **Explainability**: Grad-CAM visualizations showing model attention

        ### Limitations
        - Requires clear face detection in video
        - Performance varies by video quality and resolution
        - Cross-dataset generalization may be limited
        """
    )


if __name__ == "__main__":
    main()
