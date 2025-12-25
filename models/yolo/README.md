# YOLO Models

This directory stores YOLOv8 model files.

## Files:
- `yolov8n.pt` - PyTorch weights (downloaded automatically)
- `yolov8n.engine` - TensorRT engine (export with script)

## Export to TensorRT:

```bash
python scripts/export_yolo_tensorrt.py --model yolov8n
```

## Available models:
- yolov8n (~6MB) - nano, fastest
- yolov8s (~22MB) - small, better accuracy
- yolov8m (~50MB) - medium

## For Jetson Orin Nano 8GB:
Recommended: `yolov8n` with TensorRT FP16
