# YOLOv8 目标检测程序

这是一个基于 Ultralytics YOLOv8 的目标检测程序，可以使用自定义训练的模型进行目标检测。

## 功能特性

- 支持图片目标检测
- 支持视频目标检测
- 支持实时摄像头检测
- 可调节置信度阈值
- 自动保存检测结果

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 图片检测

```bash
python yolo_detector.py --model path/to/your/model.pt --source path/to/image.jpg
```

### 2. 视频检测

```bash
python yolo_detector.py --model path/to/your/model.pt --source path/to/video.mp4
```

### 3. 实时摄像头检测

```bash
python yolo_detector.py --model path/to/your/model.pt --source camera
```

### 参数说明

- `--model`: 必需参数，指定模型文件路径 (.pt)
- `--source`: 输入源，可以是图片/视频路径或 "camera"
- `--conf`: 置信度阈值，默认为 0.30

## 输出结果

- 图片检测结果会保存为 `result_原文件名` 的文件
- 视频检测结果会保存为 `result_原文件名` 的文件
- 检测过程中会显示实时结果窗口，按 'q' 键可退出

## 注意事项

1. 确保模型文件路径正确
2. 摄像头检测时按 'q' 键退出
3. 根据需要调整置信度阈值 (`--conf`)
