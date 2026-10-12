from pathlib import Path

import cv2
import streamlit as st

from pipeline import LicensePlateRecognizer

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = r"D:\lpr_system\models\best.pt"

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚘",
    layout="wide"
)

st.title("🚘 License Plate Recognition Demo")
st.markdown("YOLOv8 + PaddleX OCR")

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return LicensePlateRecognizer(
        model_path=MODEL_PATH,
        yolo_conf_thresh=0.45,
        use_gpu=False
    )

rec = load_model()

# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_image = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    # -----------------------------------------------------
    # Save uploaded image
    # -----------------------------------------------------

    temp_path = UPLOAD_DIR / uploaded_image.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_image.read())

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    with st.spinner("Detecting license plate..."):
        result = rec.process_image(str(temp_path))

    # -----------------------------------------------------
    # Display images
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(str(temp_path), use_container_width=True)

    with col2:
        st.subheader("Detection Result")

        ann_rgb = cv2.cvtColor(
            result.annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(ann_rgb, use_container_width=True)

    # -----------------------------------------------------
    # Detection info
    # -----------------------------------------------------

    st.divider()

    st.subheader("Detection Information")

    if not result.detections:
        st.warning("Không phát hiện biển số")
    else:
        for i, d in enumerate(result.detections, 1):

            st.success(f"Biển số #{i}: {d.plate_text}")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "YOLO Confidence",
                f"{d.yolo_conf * 100:.1f}%"
            )

            c2.metric(
                "OCR Confidence",
                f"{d.ocr_conf * 100:.1f}%"
            )

            c3.metric(
                "Latency",
                f"{d.latency_total_ms:.0f} ms"
            )

            st.code(f"BBox: {d.bbox}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("License Plate Recognition using YOLOv8 + PaddleX OCR")