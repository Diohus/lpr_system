# License Plate Recognition System
YOLOv8 detection + PaddleOCR — hỗ trợ ảnh, video, và live cam.

## Cấu trúc
```
lpr_system/
├── app/
│   ├── pipeline.py       ← core: YOLO + PaddleOCR logic
│   ├── main.py           ← FastAPI server
│   └── test_pipeline.py  ← test nhanh không cần server
├── models/
│   └── best.pt           ← ← đặt model của bạn vào đây
├── uploads/              ← file upload tạm
├── results/              ← ảnh/video đã annotate
└── requirements.txt
```

## Cài đặt

```bash
cd lpr_system
pip install -r requirements.txt
```

> **GPU (CUDA):** thay `paddlepaddle` → `paddlepaddle-gpu` trong requirements.txt

## Sử dụng

### Xử lý tất cả ảnh trong `app/uploads`

Chạy từ thư mục gốc của dự án:

```bash
python app/process_uploads.py
```

Ảnh đã đánh dấu và `summary.json` được lưu trong `results/`.

### 1. Test nhanh (không cần server)

```bash
cd app

# ảnh
python test_pipeline.py --image /path/to/car.jpg

# video (xử lý mỗi 3 frames)
python test_pipeline.py --video /path/to/traffic.mp4 --skip 2

# live webcam
python test_pipeline.py --cam 0

# tuỳ chỉnh model / threshold
python test_pipeline.py --image car.jpg --model ../models/best.pt --conf 0.5
```

### 2. Chạy API server

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập `http://localhost:8000/docs` để dùng Swagger UI.

### 3. Gọi API

```bash
# nhận diện ảnh
curl -X POST http://localhost:8000/predict/image \
  -F "file=@car.jpg"

# nhận diện video (skip 2 frames mỗi lần)
curl -X POST "http://localhost:8000/predict/video?skip_frames=2" \
  -F "file=@traffic.mp4"

# tải video kết quả
curl http://localhost:8000/results/{job_id}_annotated.mp4 -o output.mp4
```

## Tuning

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `yolo_conf_thresh` | 0.45 | Tăng nếu nhiều false positive |
| `skip_frames` (video) | 2 | Tăng để tăng FPS, giảm để tăng accuracy |
| `preprocess_crop` | adaptive threshold | Có thể thử CLAHE nếu ảnh tối |
| `use_gpu` | False | Set `USE_GPU=1` hoặc `--gpu` |

## Pipeline chi tiết

```
Input (ảnh/video/cam)
  │
  ▼
YOLOv8 (best.pt)          — phát hiện vùng biển số → bbox
  │
  ▼
Crop + Preprocess          — resize 64px height, sharpen, adaptive threshold
  │
  ▼
PaddleOCR                  — đọc ký tự → raw text + confidence
  │
  ▼
clean_plate_text()         — chuẩn hoá → format biển số VN (29A-123.45)
  │
  ▼
Output: PlateDetection     — plate_text, yolo_conf, ocr_conf, bbox, latency
```
