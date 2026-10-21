"""Local demonstration. Human review never changes thresholds automatically."""

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from vdd.pipeline import Detector, save_overlay

st.set_page_config(page_title="Visual Defect Detection", layout="wide")
st.title("Visual Defect Detection")
st.caption("Group 30 · Research prototype · Anomalies require human review")
checkpoint = st.sidebar.text_input("Checkpoint", "artifacts/baseline/model.pt")
if not Path(checkpoint).is_file():
    st.info("Train a model first using the README, then enter its checkpoint path.")
    st.stop()


@st.cache_resource
def load_detector(path, modified):
    return Detector(path)


detector = load_detector(checkpoint, Path(checkpoint).stat().st_mtime_ns)
st.sidebar.write("Product category:", detector.metadata["category"])
if not detector.metadata["pretrained"]:
    st.warning("Smoke-test model with random CNN weights. Not valid for defect evaluation.")
st.info("Upload only images of the category shown in the sidebar. Scores are not probabilities.")
upload = st.file_uploader("Product image", type=["png", "jpg", "jpeg"])
if upload:
    raw = upload.getvalue()
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
        result = detector.predict(image)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        st.error(f"Cannot process this image: {error}")
        st.stop()
    st.subheader("Anomalous" if result["anomalous"] else "Normal")
    st.write({"scores": result["scores"], "threshold": result["threshold"]})
    st.caption(
        "Heatmap is scaled separately per image: red indicates relatively high suspicion, not proven defects."
    )
    overlay_path = Path("artifacts/demo/overlay.png")
    save_overlay(image, result["heatmap"], overlay_path)
    left, right = st.columns(2)
    left.image(image, caption="Uploaded image")
    right.image(str(overlay_path), caption="Suspected regions")
    review = st.radio("Human assessment", ["Not reviewed", "Normal", "Anomalous"])
    note = st.text_input("Review note (optional)")
    if st.button("Save review", disabled=review == "Not reviewed"):
        log = Path("artifacts/demo/reviews.jsonl")
        log.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "image_sha256": hashlib.sha256(raw).hexdigest(),
            "checkpoint": checkpoint,
            "checkpoint_mtime_ns": Path(checkpoint).stat().st_mtime_ns,
            "category": detector.metadata["category"],
            "scores": result["scores"],
            "threshold": result["threshold"],
            "model_anomalous": result["anomalous"],
            "human_assessment": review,
            "note": note,
        }
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        st.success(
            "Saved locally. Threshold unchanged; use reviewed data only in a future development split."
        )
