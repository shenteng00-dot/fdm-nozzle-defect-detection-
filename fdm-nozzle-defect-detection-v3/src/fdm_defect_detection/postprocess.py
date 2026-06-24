from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, Iterable, List, Sequence, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    """Single detection result.

    Attributes:
        cls: Defect class name, such as ``stringing`` or ``nozzle_blob``.
        conf: Model confidence in [0, 1].
        bbox: Bounding box in xyxy format: (x1, y1, x2, y2).
    """

    cls: str
    conf: float
    bbox: BBox

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class Decision:
    """Final alarm / stop-printer decision."""

    alarm: bool
    stop_printer: bool
    level: str
    reason: str
    score: float
    defect_frames: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute IoU of two xyxy bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def nms(detections: Iterable[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """Class-wise non-maximum suppression."""
    det_list = list(detections)
    final_dets: List[Detection] = []
    class_names = {det.cls for det in det_list}

    for cls_name in class_names:
        cls_dets = [det for det in det_list if det.cls == cls_name]
        cls_dets = sorted(cls_dets, key=lambda det: det.conf, reverse=True)

        while cls_dets:
            best_det = cls_dets.pop(0)
            final_dets.append(best_det)

            remaining = []
            for det in cls_dets:
                iou = compute_iou(best_det.bbox, det.bbox)
                if iou < iou_threshold:
                    remaining.append(det)
            cls_dets = remaining

    return sorted(final_dets, key=lambda det: det.conf, reverse=True)


class TemporalSmoother:
    """Smooth single-frame predictions using a sliding window.

    Example: window_size=5 and min_frames=3 means alarm is stable only when at
    least 3 of the latest 5 frames contain high-confidence defects.
    """

    def __init__(self, window_size: int = 5, min_frames: int = 3, conf_threshold: float = 0.7) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_frames <= 0 or min_frames > window_size:
            raise ValueError("min_frames must be in [1, window_size]")
        self.window_size = window_size
        self.min_frames = min_frames
        self.conf_threshold = conf_threshold
        self.history: Deque[int] = deque(maxlen=window_size)

    def update(self, detections: Iterable[Detection]) -> Tuple[bool, float, int]:
        has_defect = any(det.conf >= self.conf_threshold for det in detections)
        self.history.append(1 if has_defect else 0)
        defect_frames = sum(self.history)
        temporal_score = defect_frames / self.window_size
        stable_alarm = defect_frames >= self.min_frames
        return stable_alarm, temporal_score, defect_frames


class DefectScorer:
    """Convert detections to a business risk score."""

    def __init__(self, image_width: int = 1280, image_height: int = 720) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.severity_map = {
            "stringing": 0.50,
            "nozzle_blob": 1.00,
            "support_failure": 0.90,
            "detached": 1.00,
        }

    def bbox_area_ratio(self, bbox: Sequence[float]) -> float:
        x1, y1, x2, y2 = bbox
        bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        image_area = float(self.image_width * self.image_height)
        if image_area <= 0:
            return 0.0
        return bbox_area / image_area

    def compute_score(self, det: Detection, temporal_score: float) -> float:
        severity = self.severity_map.get(det.cls, 0.50)
        area_ratio = self.bbox_area_ratio(det.bbox)
        area_score = min(1.0, area_ratio * 20.0)

        score = (
            0.50 * det.conf
            + 0.30 * severity
            + 0.15 * temporal_score
            + 0.05 * area_score
        )
        return round(min(1.0, max(0.0, score)), 3)


class DecisionMaker:
    """Map max risk score to alarm / stop-printer decision."""

    def __init__(self, alarm_threshold: float = 0.60, stop_threshold: float = 0.85) -> None:
        self.alarm_threshold = alarm_threshold
        self.stop_threshold = stop_threshold

    def decide(self, max_score: float, defect_frames: int, stable_alarm: bool) -> Decision:
        if max_score >= self.stop_threshold and stable_alarm:
            return Decision(
                alarm=True,
                stop_printer=True,
                level="high",
                reason="high_risk_nozzle_defect",
                score=round(max_score, 3),
                defect_frames=defect_frames,
            )
        if max_score >= self.alarm_threshold:
            return Decision(
                alarm=True,
                stop_printer=False,
                level="medium",
                reason="possible_printing_defect",
                score=round(max_score, 3),
                defect_frames=defect_frames,
            )
        return Decision(
            alarm=False,
            stop_printer=False,
            level="low",
            reason="normal",
            score=round(max_score, 3),
            defect_frames=defect_frames,
        )
