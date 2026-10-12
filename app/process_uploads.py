"""Process every image in app/uploads and save outputs in results."""

import json
from dataclasses import asdict
from pathlib import Path

import cv2

from pipeline import LicensePlateRecognizer


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "app" / "uploads"
OUTPUT_DIR = ROOT / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    images = sorted(
        (path for path in INPUT_DIR.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise SystemExit(f"No images found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = LicensePlateRecognizer(model_path=str(ROOT / "models" / "best.pt"))
    summary = []

    for index, image_path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
        entry = {"source": image_path.name}
        try:
            result = recognizer.process_image(str(image_path))
            output_path = OUTPUT_DIR / f"{image_path.stem}_annotated{image_path.suffix}"
            encoded, data = cv2.imencode(image_path.suffix, result.annotated_frame)
            if not encoded:
                raise RuntimeError("Could not encode annotated image")
            output_path.write_bytes(data.tobytes())
            entry.update(
                status="ok",
                annotated_image=output_path.name,
                detections=[asdict(detection) for detection in result.detections],
            )
            print(f"  {len(result.detections)} detection(s) -> {output_path.name}", flush=True)
        except Exception as exc:
            entry.update(status="error", error=str(exc))
            print(f"  ERROR: {exc}", flush=True)
        summary.append(entry)
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    succeeded = sum(item["status"] == "ok" for item in summary)
    print(f"Completed: {succeeded}/{len(images)} images. Summary: {OUTPUT_DIR / 'summary.json'}")
    if succeeded != len(images):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
