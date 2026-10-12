import cv2
import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

#fixed

@dataclass
class PlateDetection:
    plate_text: str
    yolo_conf: float
    ocr_conf: float
    bbox: list
    bbox_wh: str
    latency_yolo_ms: float
    latency_ocr_ms: float
    latency_total_ms: float
    frame_idx: Optional[int] = None


@dataclass
class FrameResult:
    frame_idx: int
    detections: list = field(default_factory=list)
    fps: Optional[float] = None
    annotated_frame: Optional[object] = None


class LicensePlateRecognizer:
    REC_MODEL = "en_PP-OCRv5_mobile_rec"

    def __init__(self, model_path="models/best.pt", yolo_conf_thresh=0.45,
                 use_gpu=False):
        self.model_path = model_path
        self.yolo_conf_thresh = yolo_conf_thresh
        self.use_gpu = use_gpu
        self._load_yolo()
        self._load_rec_model()

    def _load_yolo(self):
        from ultralytics import YOLO
        print(f"[LPR] Loading YOLO: {self.model_path}")
        self.yolo = YOLO(self.model_path)
        print("[LPR] YOLO ready!")

    def _load_rec_model(self):
        from paddlex import create_model
        device = "gpu" if self.use_gpu else "cpu"
        print(f"[LPR] Loading rec model ({self.REC_MODEL}) ...")
        self.rec = create_model(self.REC_MODEL, device=device)
        print("[LPR] PaddleX rec ready!")

    # ------------------------------------------------------------------
    # Preprocessing — nhiều chiến lược, chọn tốt nhất
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_crops(crop: np.ndarray) -> list:
        """
        Trả về list các biến thể ảnh để thử OCR.
        Với biển số 2 dòng (xe máy VN) cũng tách 2 nửa.
        """
        h, w = crop.shape[:2]
        variants = []

        def _upscale(img, target_h=48):
            sc = target_h / max(img.shape[0], 1)
            new_w = max(int(img.shape[1] * sc), target_h * 2)
            return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 1. Ảnh gốc upscale — ít xử lý nhất, thường tốt nhất cho rec model
        variants.append(("original", _upscale(crop, 96)))

        # 2. CLAHE (contrast limited adaptive histogram equalization)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eq = clahe.apply(gray)
        bgr_eq = _upscale(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR), 96)
        variants.append(("clahe", bgr_eq))

        # 3. Biển số 2 dòng (xe máy): crop nửa trên + nửa dưới riêng
        ratio = w / max(h, 1)
        if ratio < 1.6:
            mid = h // 2
            top = crop[:mid, :]
            bot = crop[mid:, :]
            if top.size > 0:
                variants.append(("top_row",    _upscale(top, 96)))
            if bot.size > 0:
                variants.append(("bottom_row", _upscale(bot, 96)))

        return variants

    @staticmethod
    def clean_plate_text(raw: str) -> str:
        text = re.sub(r"[^A-Z0-9.\-]", "", raw.upper().strip())
        m = re.match(r"^(\d{2}[A-Z]{1,2}-?)(\d{5})$", text)
        if m:
            text = m.group(1) + m.group(2)[:3] + "." + m.group(2)[3:]
        return text or "—"



    def _run_rec(self, img: np.ndarray) -> tuple[str, float]:
        """Chạy rec model trên 1 ảnh, trả (text, conf)."""
        try:
            results = list(self.rec.predict(img))
        except Exception as e:
            print(f"[LPR] rec error: {e}")
            return "", 0.0

        texts, confs = [], []
        for res in results:
            t = getattr(res, "rec_text", None)
            s = getattr(res, "rec_score", None)
            if t is None and isinstance(res, dict):
                t = res.get("rec_text") or res.get("text", "")
                s = res.get("rec_score") or res.get("score", 0.0)
            if t and str(t).strip():
                texts.append(str(t).strip())
                confs.append(float(s) if s is not None else 0.0)

        if not texts:
            return "", 0.0
        return " ".join(texts), float(np.mean(confs))

    def _ocr_best(self, crop: np.ndarray) -> tuple[str, float]:
        variants = self.prepare_crops(crop)
        candidates = []

        for name, img in variants:
            text, conf = self._run_rec(img)
            if text:
                candidates.append((name, text, conf))
                print(f"   [{name}] raw='{text}'  conf={conf:.3f}")

        if not candidates:
            return "—", 0.0

        def score(c):
            name, text, conf = c
            cleaned = self.clean_plate_text(text)
            pattern_bonus = 0.3 if re.search(r"\d{2}[A-Z]", cleaned) else 0
            return conf + pattern_bonus

        best = max(candidates, key=score)
        _, raw_text, conf = best

        top_entry = next(((n, t, c) for n, t, c in candidates if n == "top_row"), None)
        bot_entry = next(((n, t, c) for n, t, c in candidates if n == "bottom_row"), None)
        if top_entry and bot_entry:
            top_text = re.sub(r"[^A-Z0-9.\-]", "", top_entry[1].upper())
            bot_text = re.sub(r"[^A-Z0-9.\-]", "", bot_entry[1].upper())
            merged = (top_text + " " + bot_text).strip()
            avg_conf = float(np.mean([top_entry[2], bot_entry[2]]))
            print(f"   [merged] '{merged}'  conf={avg_conf:.3f}")
            return merged, avg_conf

        return self.clean_plate_text(raw_text), conf

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def detect_and_read(self, image: np.ndarray, frame_idx: int = 0) -> FrameResult:
        result = FrameResult(frame_idx=frame_idx, annotated_frame=image.copy())

        t0 = time.perf_counter()
        yolo_results = self.yolo(image, conf=self.yolo_conf_thresh, verbose=False)
        latency_yolo = (time.perf_counter() - t0) * 1000

        boxes = yolo_results[0].boxes
        if boxes is None or len(boxes) == 0:
            return result

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            yolo_conf = float(box.conf[0])

            # Padding nhỏ để không cắt sát ký tự
            pad = 4
            x1p = max(x1 - pad, 0)
            y1p = max(y1 - pad, 0)
            x2p = min(x2 + pad, image.shape[1])
            y2p = min(y2 + pad, image.shape[0])
            crop = image[y1p:y2p, x1p:x2p]

            if crop.size == 0:
                continue

            print(f"[LPR] YOLO bbox={x2-x1}x{y2-y1}  conf={yolo_conf:.3f}")

            t1 = time.perf_counter()
            plate_text, ocr_conf = self._ocr_best(crop)
            latency_ocr = (time.perf_counter() - t1) * 1000

            det = PlateDetection(
                plate_text=plate_text,
                yolo_conf=round(yolo_conf, 4),
                ocr_conf=round(ocr_conf, 4),
                bbox=[x1, y1, x2, y2],
                bbox_wh=f"{x2-x1}x{y2-y1}",
                latency_yolo_ms=round(latency_yolo, 1),
                latency_ocr_ms=round(latency_ocr, 1),
                latency_total_ms=round(latency_yolo + latency_ocr, 1),
                frame_idx=frame_idx,
            )
            result.detections.append(det)
            self._draw_box(result.annotated_frame, det)

        return result

    @staticmethod
    def _draw_box(frame: np.ndarray, det: PlateDetection):
        x1, y1, x2, y2 = det.bbox
        color = (29, 158, 117)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.plate_text}  {det.yolo_conf*100:.1f}%"
        font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), font, fs, (255, 255, 255), thick)

    # ------------------------------------------------------------------
    # High-level
    # ------------------------------------------------------------------

    def process_image(self, path: str) -> FrameResult:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Khong doc duoc anh: {path}")
        return self.detect_and_read(img, frame_idx=0)

    def process_video(self, path: str, output_path=None,
                      skip_frames: int = 0, callback=None) -> list:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Khong mo duoc video: {path}")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path,
                cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
        results, frame_idx, t_prev = [], 0, time.perf_counter()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue
                fr = self.detect_and_read(frame, frame_idx)
                t_now = time.perf_counter()
                fr.fps = round(1 / (t_now - t_prev), 1)
                t_prev = t_now
                results.append(fr)
                if writer and fr.annotated_frame is not None:
                    writer.write(fr.annotated_frame)
                if callback:
                    callback(fr)
                frame_idx += 1
        finally:
            cap.release()
            if writer:
                writer.release()
        return results
