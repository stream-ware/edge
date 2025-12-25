"""
Object Detector - Detekcja obiektów z różnych źródeł wideo

Obsługuje:
- YOLOv8 (PyTorch / TensorRT)
- Wiele kamer jednocześnie
- Event-based detection (nie każda klatka)
- Streaming detection results
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass
from collections import deque
import time

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .camera import CameraManager, CameraStream


# Tłumaczenie etykiet COCO na polski
COCO_LABELS_PL = {
    "person": "osoba", "bicycle": "rower", "car": "samochód",
    "motorcycle": "motocykl", "airplane": "samolot", "bus": "autobus",
    "train": "pociąg", "truck": "ciężarówka", "boat": "łódź",
    "traffic light": "sygnalizator", "fire hydrant": "hydrant",
    "stop sign": "znak stop", "parking meter": "parkometr",
    "bench": "ławka", "bird": "ptak", "cat": "kot", "dog": "pies",
    "horse": "koń", "sheep": "owca", "cow": "krowa",
    "elephant": "słoń", "bear": "niedźwiedź", "zebra": "zebra",
    "giraffe": "żyrafa", "backpack": "plecak", "umbrella": "parasol",
    "handbag": "torebka", "tie": "krawat", "suitcase": "walizka",
    "frisbee": "frisbee", "skis": "narty", "snowboard": "snowboard",
    "sports ball": "piłka", "kite": "latawiec", "baseball bat": "kij",
    "baseball glove": "rękawica", "skateboard": "deskorolka",
    "surfboard": "deska surfingowa", "tennis racket": "rakieta",
    "bottle": "butelka", "wine glass": "kieliszek", "cup": "kubek",
    "fork": "widelec", "knife": "nóż", "spoon": "łyżka",
    "bowl": "miska", "banana": "banan", "apple": "jabłko",
    "sandwich": "kanapka", "orange": "pomarańcza", "broccoli": "brokuł",
    "carrot": "marchewka", "hot dog": "hot dog", "pizza": "pizza",
    "donut": "pączek", "cake": "ciasto", "chair": "krzesło",
    "couch": "kanapa", "potted plant": "roślina", "bed": "łóżko",
    "dining table": "stół", "toilet": "toaleta", "tv": "telewizor",
    "laptop": "laptop", "mouse": "myszka", "remote": "pilot",
    "keyboard": "klawiatura", "cell phone": "telefon",
    "microwave": "mikrofalówka", "oven": "piekarnik", "toaster": "toster",
    "sink": "zlew", "refrigerator": "lodówka", "book": "książka",
    "clock": "zegar", "vase": "wazon", "scissors": "nożyczki",
    "teddy bear": "miś", "hair drier": "suszarka", "toothbrush": "szczoteczka",
}


@dataclass
class Detection:
    """Pojedyncze wykrycie obiektu."""
    label: str
    label_pl: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) normalized 0-1
    bbox_pixel: tuple  # (x1, y1, x2, y2) in pixels
    position: str  # "lewo-góra", "środek", etc.
    camera: str  # Source camera name
    timestamp: float


class ObjectDetector:
    """
    Detektor obiektów z obsługą wielu kamer.
    
    Przykład użycia:
        detector = ObjectDetector(config)
        await detector.initialize()
        
        # Dodaj kamery
        await detector.add_camera(0, name="usb")
        await detector.add_camera("rtsp://192.168.1.100/stream", name="ip_cam")
        
        # Stream detekcji
        async for detections in detector.stream():
            print(f"Wykryto: {[d.label_pl for d in detections]}")
    """
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("detector")
        self.config = config or {}
        
        # Model config
        self.model_name = self.config.get("model", "yolov8n")
        self.model_path = self.config.get("model_path")
        self.confidence = self.config.get("confidence", 0.5)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.max_detections = self.config.get("max_detections", 20)
        
        # Processing config
        self.process_every_n = self.config.get("process_every_n_frames", 5)
        self.translate_labels = self.config.get("translate_labels", True)
        
        # Components
        self.model: Optional[YOLO] = None
        self.camera_manager = CameraManager(self.config.get("cameras", {}))
        
        # State
        self._running = False
        self._frame_counts: Dict[str, int] = {}
        self._detection_buffer: deque = deque(maxlen=100)
    
    async def initialize(self):
        """Inicjalizacja detektora i kamer."""
        if YOLO is None:
            self.logger.error("ultralytics not installed")
            return
        
        # Load model
        self.logger.info(f"Loading model: {self.model_name}")
        
        if self.model_path:
            self.model = YOLO(self.model_path)
        else:
            self.model = YOLO(f"{self.model_name}.pt")
        
        self.logger.info("✅ Model loaded")
        
        # Initialize cameras from config
        await self.camera_manager.initialize()
    
    async def cleanup(self):
        """Zwolnienie zasobów."""
        self._running = False
        await self.camera_manager.cleanup()
        self.model = None
    
    async def add_camera(
        self,
        source,
        name: str = None,
        **kwargs
    ) -> bool:
        """
        Dodanie kamery do detektora.
        
        Args:
            source: Device ID, RTSP URL, HTTP URL, etc.
            name: Nazwa kamery
            **kwargs: Dodatkowe opcje (width, height, fps, rtsp_transport, etc.)
        
        Returns:
            True jeśli sukces
        """
        camera = await self.camera_manager.add_camera(source, name, **kwargs)
        if camera:
            self._frame_counts[camera.config.name] = 0
            return True
        return False
    
    async def remove_camera(self, name: str):
        """Usunięcie kamery."""
        await self.camera_manager.remove_camera(name)
        self._frame_counts.pop(name, None)
    
    async def stream(self) -> AsyncIterator[List[Detection]]:
        """
        Generator detekcji ze wszystkich kamer.
        
        Yields:
            Lista wykrytych obiektów
        """
        self._running = True
        
        while self._running:
            all_detections = []
            
            # Get frames from all cameras
            frames = await self.camera_manager.get_all_frames()
            
            for camera_name, frame in frames.items():
                # Process every N frames
                self._frame_counts[camera_name] = self._frame_counts.get(camera_name, 0) + 1
                
                if self._frame_counts[camera_name] % self.process_every_n != 0:
                    continue
                
                # Run detection
                detections = await self._detect(frame, camera_name)
                all_detections.extend(detections)
            
            if all_detections:
                self._detection_buffer.append({
                    "timestamp": time.time(),
                    "detections": all_detections
                })
                yield all_detections
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.01)
    
    async def detect_single(
        self, 
        camera_name: str = None,
        frame: np.ndarray = None
    ) -> List[Detection]:
        """
        Pojedyncza detekcja.
        
        Args:
            camera_name: Nazwa kamery (None = pierwsza dostępna)
            frame: Opcjonalna klatka (zamiast z kamery)
        
        Returns:
            Lista detekcji
        """
        if frame is None:
            frame = await self.camera_manager.get_frame(camera_name)
        
        if frame is None:
            return []
        
        return await self._detect(frame, camera_name or "unknown")
    
    async def _detect(self, frame: np.ndarray, camera_name: str) -> List[Detection]:
        """Wykonanie detekcji na klatce."""
        if self.model is None:
            return []
        
        loop = asyncio.get_event_loop()
        
        try:
            # Run inference in executor (YOLO is CPU-bound)
            results = await loop.run_in_executor(
                None,
                lambda: self.model(
                    frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    verbose=False
                )
            )
            
            detections = []
            h, w = frame.shape[:2]
            timestamp = time.time()
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    
                    label = result.names[cls_id]
                    label_pl = COCO_LABELS_PL.get(label, label) if self.translate_labels else label
                    
                    # Bounding box
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    bbox_norm = (
                        float(x1 / w),
                        float(y1 / h),
                        float(x2 / w),
                        float(y2 / h)
                    )
                    
                    position = self._get_position(bbox_norm)
                    
                    detections.append(Detection(
                        label=label,
                        label_pl=label_pl,
                        confidence=conf,
                        bbox=bbox_norm,
                        bbox_pixel=(int(x1), int(y1), int(x2), int(y2)),
                        position=position,
                        camera=camera_name,
                        timestamp=timestamp
                    ))
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def _get_position(self, bbox: tuple) -> str:
        """Konwersja bbox na pozycję słowną."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Horizontal
        if cx < 0.33:
            h_pos = "lewo"
        elif cx > 0.66:
            h_pos = "prawo"
        else:
            h_pos = "środek"
        
        # Vertical
        if cy < 0.33:
            v_pos = "góra"
        elif cy > 0.66:
            v_pos = "dół"
        else:
            v_pos = "środek"
        
        if h_pos == "środek" and v_pos == "środek":
            return "środek"
        elif v_pos == "środek":
            return h_pos
        elif h_pos == "środek":
            return v_pos
        else:
            return f"{v_pos}-{h_pos}"
    
    def get_current_detections(self) -> List[Detection]:
        """Pobranie ostatnich detekcji (z bufora)."""
        if self._detection_buffer:
            return self._detection_buffer[-1].get("detections", [])
        return []
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """Lista kamer."""
        return self.camera_manager.list_cameras()
    
    def format_detections_for_llm(self, detections: List[Detection] = None) -> str:
        """
        Formatowanie detekcji dla LLM.
        
        Returns:
            String opisujący wykryte obiekty
        """
        if detections is None:
            detections = self.get_current_detections()
        
        if not detections:
            return "[brak wykrytych obiektów w kadrze]"
        
        items = []
        for d in detections:
            cam_info = f" ({d.camera})" if len(self.camera_manager.cameras) > 1 else ""
            items.append(f"[{d.label_pl}: {d.position}, {d.confidence*100:.0f}%{cam_info}]")
        
        return ", ".join(items)
