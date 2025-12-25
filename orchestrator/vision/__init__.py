"""
Vision module - Camera management and object detection

Components:
- CameraManager: Multi-camera handling (USB, RTSP, HTTP, CSI)
- ObjectDetector: YOLOv8 detection
- VisionAdapter: DSL integration for orchestrator
"""

from .camera import CameraManager, CameraStream, CameraConfig, CameraType, RTSPDiscovery
from .detector import ObjectDetector, Detection
from .adapter import VisionAdapter

__all__ = [
    "CameraManager",
    "CameraStream", 
    "CameraConfig",
    "CameraType",
    "RTSPDiscovery",
    "ObjectDetector",
    "Detection",
    "VisionAdapter",
]
