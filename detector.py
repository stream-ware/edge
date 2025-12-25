"""
Object Detection module using YOLOv8 + TensorRT

Detekcja obiektów w czasie rzeczywistym
zoptymalizowana dla Jetson Orin Nano.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional, List, Dict, Any
from pathlib import Path
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# Tłumaczenie etykiet COCO na polski
COCO_PL = {
    "person": "osoba",
    "bicycle": "rower",
    "car": "samochód",
    "motorcycle": "motocykl",
    "airplane": "samolot",
    "bus": "autobus",
    "train": "pociąg",
    "truck": "ciężarówka",
    "boat": "łódź",
    "traffic light": "sygnalizator",
    "fire hydrant": "hydrant",
    "stop sign": "znak stop",
    "parking meter": "parkometr",
    "bench": "ławka",
    "bird": "ptak",
    "cat": "kot",
    "dog": "pies",
    "horse": "koń",
    "sheep": "owca",
    "cow": "krowa",
    "elephant": "słoń",
    "bear": "niedźwiedź",
    "zebra": "zebra",
    "giraffe": "żyrafa",
    "backpack": "plecak",
    "umbrella": "parasol",
    "handbag": "torebka",
    "tie": "krawat",
    "suitcase": "walizka",
    "frisbee": "frisbee",
    "skis": "narty",
    "snowboard": "snowboard",
    "sports ball": "piłka",
    "kite": "latawiec",
    "baseball bat": "kij baseballowy",
    "baseball glove": "rękawica",
    "skateboard": "deskorolka",
    "surfboard": "deska surfingowa",
    "tennis racket": "rakieta tenisowa",
    "bottle": "butelka",
    "wine glass": "kieliszek",
    "cup": "kubek",
    "fork": "widelec",
    "knife": "nóż",
    "spoon": "łyżka",
    "bowl": "miska",
    "banana": "banan",
    "apple": "jabłko",
    "sandwich": "kanapka",
    "orange": "pomarańcza",
    "broccoli": "brokuł",
    "carrot": "marchewka",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "pączek",
    "cake": "ciasto",
    "chair": "krzesło",
    "couch": "kanapa",
    "potted plant": "roślina",
    "bed": "łóżko",
    "dining table": "stół",
    "toilet": "toaleta",
    "tv": "telewizor",
    "laptop": "laptop",
    "mouse": "myszka",
    "remote": "pilot",
    "keyboard": "klawiatura",
    "cell phone": "telefon",
    "microwave": "mikrofalówka",
    "oven": "piekarnik",
    "toaster": "toster",
    "sink": "zlew",
    "refrigerator": "lodówka",
    "book": "książka",
    "clock": "zegar",
    "vase": "wazon",
    "scissors": "nożyczki",
    "teddy bear": "miś",
    "hair drier": "suszarka",
    "toothbrush": "szczoteczka",
}


class ObjectDetector:
    """
    Detektor obiektów z YOLOv8.
    
    Obsługuje:
    - YOLOv8 PyTorch
    - YOLOv8 TensorRT (zalecane dla Jetson)
    - Opcjonalny tracking (ByteTrack)
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger("vision")
        
        # Model
        self.model_name = config.get("model", "yolov8n")
        self.model_path = config.get("model_path", f"models/yolo/{self.model_name}.engine")
        
        # Detekcja
        self.confidence = config.get("confidence", 0.5)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.max_detections = config.get("max_detections", 20)
        
        # Przetwarzanie
        self.process_every_n = config.get("process_every_n_frames", 5)
        self.resolution = tuple(config.get("resolution", [640, 480]))
        self.fps = config.get("fps", 30)
        
        # Kamera
        self.camera_source = config.get("camera_source", 0)
        self.use_gstreamer = config.get("use_gstreamer", True)
        
        # Klasy
        self.classes = config.get("classes", None)
        self.translate_labels = config.get("translate_labels", True)
        
        # Tracking
        tracking_config = config.get("tracking", {})
        self.tracking_enabled = tracking_config.get("enabled", False)
        self.tracker_type = tracking_config.get("tracker", "bytetrack")
        
        # Komponenty
        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Stan
        self._running = False
        self._frame_count = 0
        self._last_detections: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Inicjalizacja modelu i kamery."""
        if cv2 is None:
            raise ImportError("opencv-python not installed")
        
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        # Sprawdź czy jest TensorRT engine
        engine_path = Path(self.model_path)
        
        if engine_path.exists() and engine_path.suffix == ".engine":
            self.logger.info(f"Ładowanie modelu TensorRT: {self.model_path}")
            self.model = YOLO(str(engine_path))
        else:
            # Fallback do PyTorch
            self.logger.info(f"Ładowanie modelu PyTorch: {self.model_name}")
            self.model = YOLO(f"{self.model_name}.pt")
        
        self.logger.info("✅ Model YOLO załadowany")
        
        # Otwórz kamerę
        await self._open_camera()
    
    async def _open_camera(self):
        """Otwarcie kamery z GStreamer (dla Jetson)."""
        if self.use_gstreamer and isinstance(self.camera_source, int):
            # GStreamer pipeline dla CSI/USB na Jetson
            gst_str = self._build_gstreamer_pipeline()
            self.logger.debug(f"GStreamer pipeline: {gst_str}")
            
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                self.logger.warning("GStreamer nie działa, fallback do V4L2")
                self.cap = cv2.VideoCapture(self.camera_source)
        else:
            self.cap = cv2.VideoCapture(self.camera_source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Nie można otworzyć kamery: {self.camera_source}")
        
        # Ustawienia
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.logger.info(f"✅ Kamera otwarta: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
    
    def _build_gstreamer_pipeline(self) -> str:
        """Budowa GStreamer pipeline dla Jetson."""
        w, h = self.resolution
        fps = self.fps
        
        # Dla CSI
        if str(self.camera_source).startswith("csi"):
            return (
                f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
        
        # Dla USB
        return (
            f"v4l2src device=/dev/video{self.camera_source} ! "
            f"video/x-raw, width={w}, height={h}, framerate={fps}/1 ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
    
    async def cleanup(self):
        """Zwolnienie zasobów."""
        self._running = False
        if self.cap:
            self.cap.release()
        self.model = None
    
    async def stream(self) -> AsyncIterator[List[Dict[str, Any]]]:
        """
        Asynchroniczny generator detekcji.
        
        Yields:
            Lista wykrytych obiektów:
            [{"label": str, "confidence": float, "bbox": (x1,y1,x2,y2)}]
        """
        self._running = True
        loop = asyncio.get_event_loop()
        
        while self._running:
            # Odczyt klatki
            ret, frame = self.cap.read()
            
            if not ret:
                self.logger.warning("Nie można odczytać klatki")
                await asyncio.sleep(0.1)
                continue
            
            self._frame_count += 1
            
            # Przetwarzaj co N klatek
            if self._frame_count % self.process_every_n != 0:
                await asyncio.sleep(0.001)  # Yield do event loop
                continue
            
            # Detekcja w executor
            detections = await loop.run_in_executor(
                None,
                self._detect,
                frame
            )
            
            self._last_detections = detections
            yield detections
    
    def _detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detekcja obiektów na klatce."""
        try:
            # Inferencja
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                classes=self.classes,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                h, w = frame.shape[:2]
                
                for i in range(len(boxes)):
                    # Klasa i pewność
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    
                    # Label
                    label = result.names[cls_id]
                    if self.translate_labels and label in COCO_PL:
                        label = COCO_PL[label]
                    
                    # Bounding box (normalized)
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    bbox_norm = (
                        float(x1 / w),
                        float(y1 / h),
                        float(x2 / w),
                        float(y2 / h)
                    )
                    
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": bbox_norm,
                        "bbox_pixel": (int(x1), int(y1), int(x2), int(y2))
                    })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Błąd detekcji: {e}")
            return []
    
    def get_current_detections(self) -> List[Dict[str, Any]]:
        """Pobranie ostatnich detekcji (bez czekania)."""
        return self._last_detections.copy()
    
    async def capture_frame(self) -> Optional[np.ndarray]:
        """Przechwycenie pojedynczej klatki."""
        ret, frame = self.cap.read()
        return frame if ret else None
    
    async def detect_single(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detekcja na pojedynczej klatce."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._detect, frame)
