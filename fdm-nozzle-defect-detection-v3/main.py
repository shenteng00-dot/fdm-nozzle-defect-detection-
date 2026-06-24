"""Minimal runnable demo without a real YOLO model.

This script simulates frame-by-frame detection outputs and runs the same
post-processing logic used in the real-time system.
"""

from src.fdm_defect_detection import Detection, DefectDetectionPipeline
from src.fdm_defect_detection.printer_control import mock_stop_printer


def build_demo_frames():
    return [
        [],
        [Detection("stringing", 0.63, (300, 200, 360, 260))],
        [
            Detection("nozzle_blob", 0.91, (160, 60, 220, 130)),
            Detection("nozzle_blob", 0.83, (158, 62, 222, 132)),  # duplicate box
        ],
        [Detection("nozzle_blob", 0.92, (162, 64, 224, 134))],
        [Detection("nozzle_blob", 0.94, (165, 65, 226, 136))],
        [Detection("support_failure", 0.81, (520, 300, 760, 500))],
    ]


def main():
    pipeline = DefectDetectionPipeline(
        image_width=1280,
        image_height=720,
        iou_threshold=0.5,
        window_size=5,
        min_frames=3,
        conf_threshold=0.7,
    )

    for frame_id, detections in enumerate(build_demo_frames(), start=1):
        output = pipeline.process(detections)
        decision = output["decision"]
        print(f"\nFrame {frame_id}")
        print("detections:", output["detections"])
        print("temporal_score:", output["temporal_score"])
        print("decision:", decision)

        if decision["stop_printer"]:
            mock_stop_printer(reason=decision["reason"])


if __name__ == "__main__":
    main()
