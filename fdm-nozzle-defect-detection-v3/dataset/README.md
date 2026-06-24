# Demo Dataset

这是一个小型合成演示数据集，主要用于让仓库结构完整，并验证 YOLOv8 的 `data.yaml`、训练脚本和推理流程可以跑通。

注意：这些图片是根据 3D 打印机喷嘴、热床、拉丝、喷嘴堆料、支撑失效、模型脱落等视觉特征绘制的 demo 样例，不等同于真实工业数据集。正式训练时应替换为真实采集和标注的数据。

目录结构：

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

类别：

```text
0 stringing
1 nozzle_blob
2 support_failure
3 detached
```
