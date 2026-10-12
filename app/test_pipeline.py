"""
Test nhanh pipeline không cần chạy API server.

Cách dùng:
  # test với ảnh
  python test_pipeline.py --image path/to/image.jpg

  # test với video
  python test_pipeline.py --video path/to/video.mp4 --skip 2

  # test với webcam (live)
  python test_pipeline.py --cam 0
"""

import argparse
import time
import cv2

from pipeline import LicensePlateRecognizer


def print_detections(frame_result, show_frame=True):
    dets = frame_result.detections
    prefix = f"[Frame {frame_result.frame_idx}]" if frame_result.frame_idx else "[Image]"
    if not dets:
        print(f"{prefix}  Không phát hiện biển số")
        return
    for d in dets:
        stars = "★" * round(d.yolo_conf * 5)
        print(
            f"{prefix}  {d.plate_text:<14}"
            f"  YOLO={d.yolo_conf*100:.1f}%"
            f"  OCR={d.ocr_conf*100:.1f}%"
            f"  bbox={d.bbox_wh}"
            f"  lat={d.latency_total_ms:.0f}ms"
            f"  {stars}"
        )


def run_image(rec, path):
    print(f"\n── Xử lý ảnh: {path}")
    result = rec.process_image(path)
    print_detections(result)

    out = path.rsplit(".", 1)[0] + "_annotated." + path.rsplit(".", 1)[-1]
    cv2.imwrite(out, result.annotated_frame)
    print(f"   Ảnh kết quả: {out}")


def run_video(rec, path, skip):
    print(f"\n── Xử lý video: {path}  (skip={skip})")
    out = path.rsplit(".", 1)[0] + "_annotated.mp4"

    def on_frame(fr):
        print_detections(fr, show_frame=True)
        if fr.fps:
            print(f"          FPS={fr.fps}")

    results = rec.process_video(path, output_path=out, skip_frames=skip, callback=on_frame)

    all_dets = [d for fr in results for d in fr.detections]
    plates   = {d.plate_text for d in all_dets if d.plate_text != "—"}
    lats     = [d.latency_total_ms for d in all_dets]
    print(f"\n── Tổng kết")
    print(f"   Frames xử lý : {len(results)}")
    print(f"   Detections   : {len(all_dets)}")
    print(f"   Biển số unique: {sorted(plates)}")
    print(f"   Latency TB   : {sum(lats)/len(lats):.1f}ms" if lats else "   (không có detection)")
    print(f"   Video output : {out}")


def run_cam(rec, cam_idx):
    print(f"\n── Live cam (index={cam_idx})  |  nhấn Q để thoát")
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("Không mở được camera")
        return

    frame_idx, t_prev = 0, time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = rec.detect_and_read(frame, frame_idx)
            print_detections(result)

            t_now = time.perf_counter()
            fps = 1 / (t_now - t_prev);  t_prev = t_now
            ann = result.annotated_frame.copy()
            cv2.putText(ann, f"FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (29, 158, 117), 2)
            cv2.imshow("LPR — press Q to quit", ann)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Plate Recognition test")
    parser.add_argument("--model", default=r"D:\lpr_system\models\best.pt")
    parser.add_argument("--conf",   type=float, default=0.45)
    parser.add_argument("--gpu",    action="store_true")
    parser.add_argument("--image",  type=str)
    parser.add_argument("--video",  type=str)
    parser.add_argument("--cam",    type=int)
    parser.add_argument("--skip",   type=int, default=2,
                        help="Bỏ qua N frames giữa mỗi lần inference (video)")
    args = parser.parse_args()

    rec = LicensePlateRecognizer(
        model_path=args.model,
        yolo_conf_thresh=args.conf,
        use_gpu=args.gpu,
    )

    if args.image:
        run_image(rec, args.image)
    elif args.video:
        run_video(rec, args.video, args.skip)
    elif args.cam is not None:
        run_cam(rec, args.cam)
    else:
        parser.print_help()
