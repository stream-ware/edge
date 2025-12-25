#!/usr/bin/env python3
"""
Export YOLOv8 to TensorRT engine for Jetson

Konwersja modelu PyTorch do TensorRT dla maksymalnej wydajności.
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics not installed")
    print("   Run: pip install ultralytics")
    sys.exit(1)


def export_tensorrt(
    model_name: str = "yolov8n",
    imgsz: int = 640,
    half: bool = True,
    output_dir: str = "models/yolo"
):
    """
    Export YOLOv8 do TensorRT.
    
    Args:
        model_name: Nazwa modelu (yolov8n, yolov8s, etc.)
        imgsz: Rozmiar wejściowy
        half: FP16 (zalecane dla Jetson)
        output_dir: Katalog wyjściowy
    """
    print("=" * 50)
    print(f"YOLOv8 → TensorRT Export")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Image size: {imgsz}")
    print(f"FP16: {half}")
    print()
    
    # Utwórz katalog
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Załaduj model PyTorch
    print(f"📥 Ładowanie {model_name}.pt...")
    model = YOLO(f"{model_name}.pt")
    
    # Export do TensorRT
    print(f"🔄 Eksport do TensorRT...")
    print("   (to może potrwać kilka minut na Jetson)")
    
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=0,  # GPU
        simplify=True,
        workspace=4,  # GB - dostosuj do RAM
    )
    
    print(f"✅ Eksport zakończony: {engine_path}")
    
    # Przenieś do output_dir
    engine_file = Path(engine_path)
    target_path = output_path / f"{model_name}.engine"
    
    if engine_file.exists() and engine_file != target_path:
        engine_file.rename(target_path)
        print(f"📁 Przeniesiono do: {target_path}")
    
    # Test
    print()
    print("🔬 Test modelu TensorRT...")
    model_trt = YOLO(str(target_path))
    
    # Dummy inference
    import numpy as np
    dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    results = model_trt(dummy_img, verbose=False)
    
    print("✅ Test OK!")
    print()
    print("=" * 50)
    print(f"Użycie w config.yaml:")
    print(f"  vision:")
    print(f"    model_path: \"{target_path}\"")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 to TensorRT for Jetson"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m"],
        help="Model name (default: yolov8n)"
    )
    parser.add_argument(
        "--imgsz", "-s",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/yolo",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    export_tensorrt(
        model_name=args.model,
        imgsz=args.imgsz,
        half=not args.fp32,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
