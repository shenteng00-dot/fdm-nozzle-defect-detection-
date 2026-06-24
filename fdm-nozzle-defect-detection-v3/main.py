"""
FDM 3D 打印喷嘴堵头与拉丝异常实时视觉检测系统

正式运行入口：
1. 支持摄像头、视频文件、图片文件输入
2. 使用 YOLOv8 模型进行异常检测
3. 将 YOLO 输出转换为项目内部 Detection 格式
4. 执行 NMS 去重、时序平滑、异常 score 计算
5. 根据风险等级输出报警或停机决策
"""

import argparse
import time
from pathlib import Path
from typing import List, Union

import cv2
from ultralytics import YOLO

from src.fdm_defect_detection import Detection, DefectDetectionPipeline
from src.fdm_defect_detection.printer_control import (
    mock_stop_printer,
    send_emergency_stop,
)


CLASS_NAMES = {
    0: "stringing",
    1: "nozzle_blob",
    2: "support_failure",
    3: "detached",
}


def parse_args():
    """
    解析命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="FDM 3D 打印喷嘴堵头与拉丝异常实时视觉检测系统"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="YOLOv8 模型权重路径，例如 models/best.pt",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="输入源。0 表示默认摄像头，也可以是图片路径或视频路径",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="YOLO 检测置信度阈值",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU 阈值",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="时序平滑窗口大小",
    )

    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="窗口内至少多少帧出现异常才认为异常稳定",
    )

    parser.add_argument(
        "--alarm-conf",
        type=float,
        default=0.7,
        help="参与报警判断的检测置信度阈值",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="是否显示实时检测画面",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="是否保存检测结果视频",
    )

    parser.add_argument(
        "--stop-mode",
        type=str,
        default="mock",
        choices=["mock", "serial", "none"],
        help="停机模式：mock 表示模拟停机，serial 表示串口停机，none 表示不执行停机",
    )

    parser.add_argument(
        "--serial-port",
        type=str,
        default="/dev/ttyUSB0",
        help="串口停机时使用的端口",
    )

    return parser.parse_args()


def is_camera_source(source: str) -> bool:
    """
    判断输入源是否为摄像头编号。
    """

    return source.isdigit()


def yolo_results_to_detections(results) -> List[Detection]:
    """
    将 YOLOv8 的输出结果转换为项目内部 Detection 格式。

    Detection 格式：
    Detection(cls_name, confidence, bbox)

    bbox 格式：
    (x1, y1, x2, y2)
    """

    detections = []

    if results.boxes is None:
        return detections

    boxes = results.boxes

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )

        cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

        detections.append(
            Detection(
                cls=cls_name,
                conf=conf,
                bbox=bbox,
            )
        )

    return detections


def draw_results(frame, detections, decision):
    """
    在图像上绘制检测框和决策结果。
    """

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        label = f"{det.cls} {det.conf:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    alarm_text = f"alarm: {decision['alarm']} | stop: {decision['stop_printer']} | level: {decision['level']}"

    cv2.putText(
        frame,
        alarm_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    if "score" in decision:
        score_text = f"score: {decision['score']:.3f}"
        cv2.putText(
            frame,
            score_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    return frame


def execute_stop_action(args, reason: str):
    """
    根据设置执行停机动作。
    """

    if args.stop_mode == "none":
        print("[INFO] Stop action skipped.")
        return

    if args.stop_mode == "mock":
        mock_stop_printer(reason=reason)
        return

    if args.stop_mode == "serial":
        send_emergency_stop(
            port=args.serial_port,
            reason=reason,
        )
        return


def process_frame(model, pipeline, frame, args):
    """
    处理单帧图像：
    1. YOLO 推理
    2. 结果格式转换
    3. 后处理流水线
    4. 返回输出结果
    """

    results = model.predict(
        source=frame,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )[0]

    raw_detections = yolo_results_to_detections(results)

    output = pipeline.process(raw_detections)

    return output


def run_image(model, pipeline, image_path: str, args):
    """
    处理单张图片。
    """

    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    output = process_frame(model, pipeline, frame, args)
    decision = output["decision"]

    print("detections:", output["detections"])
    print("temporal_score:", output["temporal_score"])
    print("decision:", decision)

    frame = draw_results(frame, output["detections"], decision)

    if args.save:
        save_dir = Path("demo/result_images")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / Path(image_path).name
        cv2.imwrite(str(save_path), frame)

        print(f"[INFO] Result saved to {save_path}")

    if args.show:
        cv2.imshow("FDM Defect Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if decision["stop_printer"]:
        execute_stop_action(args, reason=decision["reason"])


def run_video_or_camera(model, pipeline, source: Union[str, int], args):
    """
    处理摄像头或视频流。
    """

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入源：{source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    writer = None

    if args.save:
        save_dir = Path("demo/result_videos")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(save_path),
            fourcc,
            fps,
            (width, height),
        )

        print(f"[INFO] Video result will be saved to {save_path}")

    frame_id = 0
    stop_triggered = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1
        start_time = time.time()

        output = process_frame(model, pipeline, frame, args)
        decision = output["decision"]

        cost_time = time.time() - start_time
        fps_now = 1.0 / cost_time if cost_time > 0 else 0.0

        print(
            f"Frame {frame_id} | "
            f"detections={len(output['detections'])} | "
            f"temporal_score={output['temporal_score']:.3f} | "
            f"decision={decision} | "
            f"fps={fps_now:.2f}"
        )

        frame = draw_results(frame, output["detections"], decision)

        cv2.putText(
            frame,
            f"FPS: {fps_now:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("FDM Defect Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if decision["stop_printer"] and not stop_triggered:
            execute_stop_action(args, reason=decision["reason"])
            stop_triggered = True

    cap.release()

    if writer is not None:
        writer.release()

    if args.show:
        cv2.destroyAllWindows()


def main():
    """
    主函数。
    """

    args = parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在：{model_path}\n"
            f"请将训练好的模型放到 models/ 目录下，例如 models/best.pt"
        )

    print("[INFO] Loading YOLO model...")
    model = YOLO(str(model_path))

    source = args.source

    if is_camera_source(source):
        input_source = int(source)
    else:
        input_source = source

    pipeline = DefectDetectionPipeline(
        image_width=1280,
        image_height=720,
        iou_threshold=args.iou,
        window_size=args.window_size,
        min_frames=args.min_frames,
        conf_threshold=args.alarm_conf,
    )

    print("[INFO] FDM defect detection system started.")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Source: {args.source}")

    if isinstance(input_source, str):
        suffix = Path(input_source).suffix.lower()

        if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            run_image(model, pipeline, input_source, args)
        else:
            run_video_or_camera(model, pipeline, input_source, args)
    else:
        run_video_or_camera(model, pipeline, input_source, args)

    print("[INFO] Detection finished.")


if __name__ == "__main__":
    main()
