"""YOLOv8 image prediction entry.

Usage:
    python predict.py --model models/best.pt --source demo/test_images --save-dir demo/result_images
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.fdm_defect_detection import Detection, DefectDetectionPipeline


CLASS_NAMES = ["stringing", "nozzle_blob", "support_failure", "detached"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt", help="YOLOv8 model path")
    parser.add_argument("--source", default="demo/test_images", help="Image / directory / video source")
    parser.add_argument("--save-dir", default="demo/result_images", help="Directory to save visualization results")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    return parser.parse_args()


def yolo_results_to_detections(result) -> List[Detection]:
    detections: List[Detection] = []
    if result.boxes is None:
        return detections
    boxes = result.boxes
    for xyxy, conf, cls_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
        cls_index = int(cls_id)
        cls_name = CLASS_NAMES[cls_index] if 0 <= cls_index < len(CLASS_NAMES) else str(cls_index)
        detections.append(Detection(cls=cls_name, conf=float(conf), bbox=tuple(map(float, xyxy))))
    return detections


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install it with: pip install ultralytics") from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Put your trained model in models/best.pt")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    pipeline = DefectDetectionPipeline()

    results = model.predict(source=args.source, conf=args.conf, save=True, project=str(save_dir), name="predict", exist_ok=True)
    for idx, result in enumerate(results):
        detections = yolo_results_to_detections(result)
        output = pipeline.process(detections)
        print(f"\nImage/Frame {idx}")
        print(output)


if __name__ == "__main__":
    main()
