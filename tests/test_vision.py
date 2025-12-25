"""
Tests for Vision/Object Detection module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestObjectDetector:
    """Tests for ObjectDetector module."""
    
    @pytest.fixture
    def vision_config(self):
        return {
            "model": "yolov8n",
            "model_path": "models/yolo/yolov8n.pt",
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "max_detections": 20,
            "process_every_n_frames": 5,
            "resolution": [640, 480],
            "fps": 30,
            "camera_source": 0,
            "use_gstreamer": False,
            "translate_labels": True,
            "tracking": {
                "enabled": False
            }
        }
    
    def test_config_parsing(self, vision_config):
        """Test configuration parsing."""
        from src.vision.detector import ObjectDetector
        
        detector = ObjectDetector(vision_config)
        
        assert detector.model_name == "yolov8n"
        assert detector.confidence == 0.5
        assert detector.resolution == (640, 480)
        assert detector.process_every_n == 5
    
    def test_position_calculation(self, vision_config):
        """Test bbox to position conversion."""
        from src.vision.detector import ObjectDetector
        
        detector = ObjectDetector(vision_config)
        
        # Center of frame
        pos = detector._get_position((0.33, 0.33, 0.66, 0.66))
        assert pos == "środek"
        
        # Top-left
        pos = detector._get_position((0.0, 0.0, 0.2, 0.2))
        assert "góra" in pos or "lewo" in pos
        
        # Bottom-right
        pos = detector._get_position((0.8, 0.8, 1.0, 1.0))
        assert "dół" in pos or "prawo" in pos
    
    def test_label_translation(self, vision_config):
        """Test COCO label translation to Polish."""
        from src.vision.detector import COCO_PL
        
        assert COCO_PL["person"] == "osoba"
        assert COCO_PL["cup"] == "kubek"
        assert COCO_PL["laptop"] == "laptop"
        assert COCO_PL["keyboard"] == "klawiatura"


class TestGStreamerPipeline:
    """Tests for GStreamer pipeline generation."""
    
    @pytest.fixture
    def vision_config(self):
        return {
            "model": "yolov8n",
            "resolution": [640, 480],
            "fps": 30,
            "camera_source": 0,
            "use_gstreamer": True,
        }
    
    def test_usb_camera_pipeline(self, vision_config):
        """Test USB camera GStreamer pipeline."""
        from src.vision.detector import ObjectDetector
        
        detector = ObjectDetector(vision_config)
        pipeline = detector._build_gstreamer_pipeline()
        
        assert "v4l2src" in pipeline
        assert "640" in pipeline
        assert "480" in pipeline
        assert "BGR" in pipeline
