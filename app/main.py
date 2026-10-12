"""
FastAPI backend — License Plate Recognition
Endpoints:
  POST /predict/image   — upload 1 ảnh
  POST /predict/video   — upload 1 video
  GET  /results/{id}    — lấy kết quả theo job id
  GET  /health
"""

import os, uuid, shutil, asyncio
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from pipeline import LicensePlateRecognizer, FrameResult

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"D:\lpr_system\models\best.pt"
)
YOLO_CONF    = float(os.getenv("YOLO_CONF", "0.45"))
USE_GPU      = os.getenv("USE_GPU", "0") == "1"
UPLOAD_DIR   = Path("uploads");  UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR  = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------
app = FastAPI(
    title="License Plate Recognition API",
    description="YOLOv8 + PaddleOCR pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Lazy-load model (chỉ load 1 lần khi khởi động)
# ---------------------------------------------------------------
_recognizer: Optional[LicensePlateRecognizer] = None

def get_recognizer() -> LicensePlateRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = LicensePlateRecognizer(
            model_path=MODEL_PATH,
            yolo_conf_thresh=YOLO_CONF,
            use_gpu=USE_GPU,
        )
    return _recognizer


# ---------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------
class DetectionOut(BaseModel):
    plate_text: str
    yolo_conf: float
    ocr_conf: float
    bbox: list[int]
    bbox_wh: str
    latency_yolo_ms: float
    latency_ocr_ms: float
    latency_total_ms: float
    frame_idx: Optional[int] = None

class FrameOut(BaseModel):
    frame_idx: int
    fps: Optional[float]
    detections: list[DetectionOut]

class ImageResponse(BaseModel):
    job_id: str
    source: str
    total_detections: int
    avg_yolo_conf: float
    avg_ocr_conf: float
    avg_latency_ms: float
    annotated_image_url: str
    detections: list[DetectionOut]

class VideoResponse(BaseModel):
    job_id: str
    source: str
    status: str                   # "processing" | "done"
    total_frames: int
    total_detections: int
    unique_plates: list[str]
    avg_fps: float
    avg_latency_ms: float
    annotated_video_url: Optional[str]
    frames: list[FrameOut]


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------
def _save_upload(upload: UploadFile, dest: Path):
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)

def _frame_result_to_out(fr: FrameResult) -> FrameOut:
    return FrameOut(
        frame_idx=fr.frame_idx,
        fps=fr.fps,
        detections=[DetectionOut(**d.__dict__) for d in fr.detections],
    )

def _safe_avg(vals):
    return round(float(np.mean(vals)), 2) if vals else 0.0


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict/image", response_model=ImageResponse)
async def predict_image(file: UploadFile = File(...)):
    # validate
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File phải là ảnh (image/*)")

    job_id  = uuid.uuid4().hex[:10]
    suffix  = Path(file.filename).suffix or ".jpg"
    in_path = UPLOAD_DIR / f"{job_id}{suffix}"
    _save_upload(file, in_path)

    rec = get_recognizer()
    try:
        frame_result = rec.process_image(str(in_path))
    except Exception as e:
        raise HTTPException(500, f"Lỗi xử lý: {e}")

    # lưu ảnh annotated
    ann_path = RESULTS_DIR / f"{job_id}_annotated{suffix}"
    cv2.imwrite(str(ann_path), frame_result.annotated_frame)

    dets = frame_result.detections
    return ImageResponse(
        job_id=job_id,
        source=file.filename,
        total_detections=len(dets),
        avg_yolo_conf=_safe_avg([d.yolo_conf for d in dets]),
        avg_ocr_conf=_safe_avg([d.ocr_conf for d in dets]),
        avg_latency_ms=_safe_avg([d.latency_total_ms for d in dets]),
        annotated_image_url=f"/results/{job_id}_annotated{suffix}",
        detections=[DetectionOut(**d.__dict__) for d in dets],
    )


@app.post("/predict/video", response_model=VideoResponse)
async def predict_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    skip_frames: int = 2,          # xử lý mỗi 3 frames (tiết kiệm tài nguyên)
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File phải là video (video/*)")

    job_id  = uuid.uuid4().hex[:10]
    suffix  = Path(file.filename).suffix or ".mp4"
    in_path = UPLOAD_DIR / f"{job_id}{suffix}"
    _save_upload(file, in_path)

    out_path = RESULTS_DIR / f"{job_id}_annotated.mp4"

    rec = get_recognizer()
    frame_results = rec.process_video(
        str(in_path),
        output_path=str(out_path),
        skip_frames=skip_frames,
    )

    all_dets  = [d for fr in frame_results for d in fr.detections]
    unique    = list({d.plate_text for d in all_dets if d.plate_text != "—"})
    fps_vals  = [fr.fps for fr in frame_results if fr.fps]

    return VideoResponse(
        job_id=job_id,
        source=file.filename,
        status="done",
        total_frames=len(frame_results),
        total_detections=len(all_dets),
        unique_plates=unique,
        avg_fps=_safe_avg(fps_vals),
        avg_latency_ms=_safe_avg([d.latency_total_ms for d in all_dets]),
        annotated_video_url=f"/results/{job_id}_annotated.mp4",
        frames=[_frame_result_to_out(fr) for fr in frame_results],
    )


@app.get("/results/{filename}")
def get_result_file(filename: str):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File không tồn tại")
    return FileResponse(str(path))
