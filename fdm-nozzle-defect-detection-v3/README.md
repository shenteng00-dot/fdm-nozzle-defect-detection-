# FDM 3D 打印喷嘴堵头与拉丝异常实时视觉检测系统

## 项目简介

本项目面向 FDM 3D 打印过程中的在线状态监控，主要用于实时检测喷嘴堵头、拉丝、支撑失效、模型脱落等异常情况。

系统基于 YOLOv8 目标检测框架，对摄像头采集到的打印过程图像进行推理，并在检测结果后加入 NMS 去重、时序平滑、异常 score 计算和报警停机决策逻辑，从而降低单帧误检带来的误报警风险。

本仓库是一个本地复现 demo，重点展示完整的视觉检测流程和工程实现思路。

## 检测类别

| 类别 | 含义 | 风险说明 |
|---|---|---|
| `stringing` | 拉丝 / 牵丝 | 喷头移动过程中产生细丝，影响成型质量 |
| `nozzle_blob` | 喷嘴堆料 / 堵头 | 材料在喷嘴附近异常堆积，风险较高 |
| `support_failure` | 支撑失效 | 支撑结构异常，可能导致模型塌陷 |
| `detached` | 模型脱落 | 模型与热床分离，容易造成连续打印失败 |

## 技术路线

```text
视频流采集
    ↓
图像预处理
    ↓
YOLOv8s 目标检测
    ↓
NMS 去除重复框
    ↓
时序平滑
    ↓
异常 score 计算
    ↓
报警 / 停机决策
```

## 项目结构

```text
fdm-nozzle-defect-detection/
├── README.md
├── requirements.txt
├── data.yaml
├── main.py
├── predict.py
├── train.py
├── stop_printer.py
├── configs/
│   └── default.yaml
├── src/
│   └── fdm_defect_detection/
│       ├── __init__.py
│       ├── postprocess.py
│       ├── pipeline.py
│       └── printer_control.py
├── docs/
│   ├── 项目说明.md
│   ├── 面试讲解稿.md
│   └── 流程图.md
├── demo/
│   ├── test_images/
│   └── result_images/
├── models/
└── tests/
    └── test_postprocess.py
```

## 快速运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行不依赖真实模型的最小 demo：

```bash
python main.py
```

该 demo 会模拟连续帧检测结果，并输出 NMS、时序平滑、风险 score 和报警停机决策。

## YOLOv8 推理

将训练好的模型放到：

```text
models/best.pt
```

运行：

```bash
python predict.py --model models/best.pt --source demo/test_images --save-dir demo/result_images
```

## 模型训练

数据集路径在 `data.yaml` 中配置：

```yaml
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test

nc: 4

names:
  0: stringing
  1: nozzle_blob
  2: support_failure
  3: detached
```

运行训练：

```bash
python train.py --data data.yaml --epochs 100 --imgsz 640 --model yolov8s.pt
```

或使用 YOLO 命令：

```bash
yolo detect train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

## 后处理逻辑

### NMS 去重

模型可能对同一个异常输出多个框，NMS 会保留置信度最高的框，并删除 IoU 过高的重复框。

### 时序平滑

单帧误检可能由反光、模糊、遮挡引起。项目使用滑动窗口判断，例如最近 5 帧中至少 3 帧出现高置信异常，才认为异常较稳定。

### 异常 score

风险分数综合考虑：

- 模型置信度 `conf`
- 异常类别严重程度 `severity`
- 连续帧稳定性 `temporal_score`
- 异常区域面积 `area_score`

示例公式：

```text
score = 0.50 * conf + 0.30 * severity + 0.15 * temporal_score + 0.05 * area_score
```

### 报警停机决策

| score 范围 | 决策 |
|---|---|
| `score >= 0.85` 且连续帧稳定 | 报警并停机 |
| `0.60 <= score < 0.85` | 只报警 |
| `score < 0.60` | 继续打印 |

## 注意事项

本仓库不直接上传数据集、训练结果和模型权重。相关文件已在 `.gitignore` 中排除：

```text
dataset/
runs/
wandb/
*.pt
*.onnx
*.rknn
```

## 面试介绍参考

这个项目主要是面向 FDM 3D 打印过程中的在线异常监控。我使用 YOLOv8s 对打印过程中的拉丝、喷嘴堆料、支撑失效和模型脱落等异常进行检测。项目重点不只是单帧目标检测，而是在检测结果后加入了 NMS 去重、连续帧时序平滑、异常 score 计算和报警停机逻辑。

其中，NMS 用来去除同一异常目标的重复框；时序平滑用于避免单帧误检导致误停机；异常 score 综合考虑置信度、异常类别严重程度、连续帧稳定性和异常面积，最终输出是否报警或停机。整体目标是把视觉检测结果转化为工程上可用的自动监控决策。

## Demo 数据集说明

本仓库包含一个小型合成 demo 数据集，位于：

```text
dataset/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
└── labels/test
```

该数据集包含 `stringing`、`nozzle_blob`、`support_failure`、`detached` 四类样例，主要用于展示 YOLOv8 数据集组织方式、`data.yaml` 配置方式和训练脚本运行流程。需要注意的是，demo 数据集是合成样例，不代表真实工业现场数据。正式项目训练时，应替换为真实采集并标注的数据集。

如果需要重新生成 demo 数据集，可以运行：

```bash
python tools/generate_demo_dataset.py
```
