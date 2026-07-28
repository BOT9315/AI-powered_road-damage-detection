"""
app.py
======
Streamlit demo application for AI-Powered Road Damage Detection.

Run with:
    streamlit run app/app.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `src` imports

from src.model import RoadDamageModel
from src.utils import draw_detections, load_config, summarize_detections

st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="wide")

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🛣️ Road Damage Detection")
st.sidebar.markdown("AI-powered pothole & crack detection using computer vision (YOLOv8).")

cfg = load_config("config.yaml")
class_names = cfg["dataset"]["classes"]

weights_path = st.sidebar.text_input("Model weights path", value=cfg["inference"]["weights_path"])
conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, cfg["inference"]["conf_threshold"], 0.05)
iou_thresh = st.sidebar.slider("IoU threshold (NMS)", 0.05, 0.95, cfg["inference"]["iou_threshold"], 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Detected damage classes:**")
for cid, cname in class_names.items():
    st.sidebar.markdown(f"- `{cid}` {cname}")


@st.cache_resource(show_spinner="Loading model...")
def get_model(weights, device):
    return RoadDamageModel(weights=weights, device=device)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
st.title("AI-Powered Road Damage Detection")
st.caption("Upload a road image (or video) to automatically detect potholes, cracks, "
           "and faded markings, with estimated severity.")

tab_image, tab_video, tab_batch = st.tabs(["📷 Single Image", "🎞️ Video", "📁 Batch Images"])

# ---- Single image tab ----
with tab_image:
    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"], key="single")
    if uploaded is not None:
        col1, col2 = st.columns(2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        image = cv2.imread(tmp_path)
        col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

        if st.button("Run Detection", key="run_single"):
            try:
                model = get_model(weights_path, cfg["project"]["device"])
                results = model.predict(tmp_path, conf=conf_thresh, iou=iou_thresh,
                                         img_size=cfg["dataset"]["img_size"])
                detections = RoadDamageModel.parse_results(results, class_names)
                annotated = draw_detections(image, detections, class_names)
                col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                           caption="Detections", use_container_width=True)

                summary = summarize_detections(detections, class_names)
                st.subheader("Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Detections", summary["total_detections"])
                m2.metric("High Severity", summary["by_severity"]["High"])
                m3.metric("Critical Severity", summary["by_severity"]["Critical"])
                m4.metric("Damage Types Found", len(summary["by_class"]))

                if detections:
                    df = pd.DataFrame(detections)[["class_name", "confidence", "severity", "box"]]
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No damage detected above the current confidence threshold.")
            except Exception as e:
                st.error(f"Inference failed: {e}\n\nMake sure a trained model exists at "
                          f"the weights path (see README for training instructions).")

# ---- Video tab ----
with tab_video:
    uploaded_video = st.file_uploader("Upload road video", type=["mp4", "avi", "mov"], key="video")
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.video(video_path)
        if st.button("Run Detection on Video", key="run_video"):
            st.warning("Video processing runs frame-by-frame and may take a while for long clips.")
            try:
                model = get_model(weights_path, cfg["project"]["device"])
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                w, h = int(cap.get(3)), int(cap.get(4))
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

                progress = st.progress(0)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                frame_idx = 0
                all_detections = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model.predict(frame, conf=conf_thresh, iou=iou_thresh,
                                             img_size=cfg["dataset"]["img_size"])
                    detections = RoadDamageModel.parse_results(results, class_names)
                    all_detections.extend(detections)
                    writer.write(draw_detections(frame, detections, class_names))
                    frame_idx += 1
                    progress.progress(min(frame_idx / total, 1.0))

                cap.release()
                writer.release()

                st.success("Processing complete.")
                st.video(out_path)
                summary = summarize_detections(all_detections, class_names)
                st.json(summary)
            except Exception as e:
                st.error(f"Video inference failed: {e}")

# ---- Batch tab ----
with tab_batch:
    uploaded_files = st.file_uploader("Upload multiple road images", type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True, key="batch")
    if uploaded_files and st.button("Run Batch Detection"):
        model = get_model(weights_path, cfg["project"]["device"])
        rows = []
        for uf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            image = cv2.imread(tmp_path)
            results = model.predict(tmp_path, conf=conf_thresh, iou=iou_thresh,
                                     img_size=cfg["dataset"]["img_size"])
            detections = RoadDamageModel.parse_results(results, class_names)
            summary = summarize_detections(detections, class_names)
            rows.append({
                "filename": uf.name,
                "total_detections": summary["total_detections"],
                "high_or_critical": summary["by_severity"]["High"] + summary["by_severity"]["Critical"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.caption("Built with YOLOv8 (Ultralytics) + Streamlit · See README.md for training your own model.")


#cd "C:\Users\akrk0\Downloads\road-damage-detection (1)\road-damage-detection"
#venv\Scripts\Activate.ps1
#streamlit run app/app.py
