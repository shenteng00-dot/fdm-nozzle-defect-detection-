"""YOLOv8 training template.

Usage:
    python train.py --data data.yaml --epochs 100 --imgsz 640 --model yolov8s.pt
"""

from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="fdm_nozzle_yolov8s")
    return parser.parse_args()


def main():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install it with: pip install ultralytics") from exc

    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
