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
            "translate_labels": True,
        }
    
    def test_config_parsing(self, vision_config):
        """Test configuration parsing."""
        from orchestrator.vision.detector import ObjectDetector
        
        detector = ObjectDetector(vision_config)
        
        assert detector.model_name == "yolov8n"
        assert detector.confidence == 0.5
        assert detector.process_every_n == 5
    
    def test_position_calculation(self, vision_config):
        """Test bbox to position conversion."""
        from orchestrator.vision.detector import ObjectDetector
        
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
        from orchestrator.vision.detector import COCO_LABELS_PL
        
        assert COCO_LABELS_PL["person"] == "osoba"
        assert COCO_LABELS_PL["cup"] == "kubek"
        assert COCO_LABELS_PL["laptop"] == "laptop"
        assert COCO_LABELS_PL["keyboard"] == "klawiatura"


class TestCameraManager:
    """Tests for CameraManager module."""
    
    @pytest.fixture
    def camera_config(self):
        return {
            "cameras": []
        }
    
    def test_camera_manager_init(self, camera_config):
        """Test CameraManager initialization."""
        from orchestrator.vision.camera import CameraManager
        
        manager = CameraManager(camera_config)
        
        assert len(manager.cameras) == 0
    
    def test_camera_list(self, camera_config):
        """Test camera listing."""
        from orchestrator.vision.camera import CameraManager
        
        manager = CameraManager(camera_config)
        cameras = manager.list_cameras()
        
        assert isinstance(cameras, list)
        assert len(cameras) == 0
