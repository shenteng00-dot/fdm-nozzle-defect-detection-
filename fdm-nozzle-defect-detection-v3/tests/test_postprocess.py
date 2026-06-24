from src.fdm_defect_detection import Detection, DefectDetectionPipeline, compute_iou, nms


def test_iou():
    box1 = (10, 10, 50, 50)
    box2 = (30, 30, 70, 70)
    assert round(compute_iou(box1, box2), 4) == 0.1429


def test_nms():
    dets = [
        Detection("nozzle_blob", 0.92, (160, 60, 220, 130)),
        Detection("nozzle_blob", 0.85, (158, 62, 222, 132)),
        Detection("stringing", 0.76, (300, 200, 360, 260)),
    ]
    filtered = nms(dets, iou_threshold=0.5)
    assert len(filtered) == 2


def test_pipeline_decision():
    pipeline = DefectDetectionPipeline(window_size=3, min_frames=2, conf_threshold=0.7)
    det = Detection("nozzle_blob", 0.95, (160, 60, 220, 130))
    pipeline.process([det])
    output = pipeline.process([det])
    assert output["decision"]["alarm"] is True
