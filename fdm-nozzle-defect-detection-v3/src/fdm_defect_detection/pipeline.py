from __future__ import annotations

from typing import Dict, Iterable, List

from .postprocess import Detection, Decision, DecisionMaker, DefectScorer, TemporalSmoother, nms


class DefectDetectionPipeline:
    """Post-processing pipeline for FDM 3D printer defect detection."""

    def __init__(
        self,
        image_width: int = 1280,
        image_height: int = 720,
        iou_threshold: float = 0.5,
        window_size: int = 5,
        min_frames: int = 3,
        conf_threshold: float = 0.7,
        alarm_threshold: float = 0.60,
        stop_threshold: float = 0.85,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.smoother = TemporalSmoother(window_size, min_frames, conf_threshold)
        self.scorer = DefectScorer(image_width, image_height)
        self.decision_maker = DecisionMaker(alarm_threshold, stop_threshold)

    def process(self, detections: Iterable[Detection]) -> Dict[str, object]:
        filtered = nms(detections, self.iou_threshold)
        stable_alarm, temporal_score, defect_frames = self.smoother.update(filtered)

        scored_results: List[Dict[str, object]] = []
        max_score = 0.0
        for det in filtered:
            score = self.scorer.compute_score(det, temporal_score)
            result = det.to_dict()
            result["score"] = score
            scored_results.append(result)
            max_score = max(max_score, score)

        decision: Decision = self.decision_maker.decide(max_score, defect_frames, stable_alarm)
        return {
            "detections": scored_results,
            "temporal_score": round(temporal_score, 3),
            "decision": decision.to_dict(),
        }
