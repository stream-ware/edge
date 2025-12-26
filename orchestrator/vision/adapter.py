"""
Vision Adapter - Obsługa komend wizyjnych przez DSL

Obsługuje:
- Dodawanie/usuwanie kamer
- Zapytania o wykryte obiekty
- Sterowanie detekcją
- Streaming events
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from .camera import CameraManager, RTSPDiscovery
from .detector import ObjectDetector, Detection


class VisionAdapter:
    """
    Adapter wizyjny dla Orchestratora.
    
    Obsługuje komendy DSL:
    - vision.add_camera
    - vision.remove_camera
    - vision.list_cameras
    - vision.detect
    - vision.describe
    - vision.scan_network
    """
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("vision_adapter")
        self.config = config or {}

        self.detector: Optional[ObjectDetector] = None
        self._initialized = False
        self._running = False
        self._detection_task: Optional[asyncio.Task] = None
        
        # Callbacks dla eventów
        self._on_detection_callbacks = []
    
    async def initialize(self):
        """Inicjalizacja adaptera."""
        if self._initialized:
            return

        self.detector = ObjectDetector(self.config)
        await self.detector.initialize()
        
        # Auto-add cameras from config
        cameras_config = self.config.get("cameras", [])
        for cam in cameras_config:
            await self.detector.add_camera(
                source=cam.get("source"),
                name=cam.get("name"),
                **{k: v for k, v in cam.items() if k not in ["source", "name"]}
            )
        
        # Start detection loop
        self._running = True
        self._detection_task = asyncio.create_task(self._detection_loop())

        self._initialized = True
        
        self.logger.info("✅ Vision adapter initialized")
    
    async def cleanup(self):
        """Zwolnienie zasobów."""
        self._running = False
        if self._detection_task:
            self._detection_task.cancel()
        if self.detector:
            await self.detector.cleanup()
        self._initialized = False
    
    async def _detection_loop(self):
        """Background detection loop."""
        if not self.detector:
            return
        try:
            async for detections in self.detector.stream():
                for callback in self._on_detection_callbacks:
                    try:
                        await callback(detections)
                    except Exception as e:
                        self.logger.error(f"Detection callback error: {e}")
        except asyncio.CancelledError:
            pass
    
    def on_detection(self, callback):
        """Rejestracja callback dla nowych detekcji."""
        self._on_detection_callbacks.append(callback)
    
    async def execute(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wykonanie akcji wizyjnej.
        
        Obsługiwane akcje:
        - vision.add_camera: Dodanie kamery
        - vision.remove_camera: Usunięcie kamery
        - vision.list_cameras: Lista kamer
        - vision.detect: Wykonaj detekcję
        - vision.describe: Opisz co widać
        - vision.scan_network: Skanuj sieć RTSP
        """
        action = dsl.get("action", "")

        if not self._initialized:
            await self.initialize()
        
        handlers = {
            "vision.add_camera": self._add_camera,
            "vision.remove_camera": self._remove_camera,
            "vision.list_cameras": self._list_cameras,
            "vision.detect": self._detect,
            "vision.describe": self._describe,
            "vision.scan_network": self._scan_network,
            "vision.count": self._count_objects,
            "vision.find": self._find_object,
        }
        
        handler = handlers.get(action)
        
        if not handler:
            return {"status": "error", "error": f"Unknown action: {action}"}
        
        return await handler(dsl)
    
    async def _add_camera(self, dsl: dict) -> dict:
        """Dodanie kamery."""
        if not self.detector:
            return {"status": "error", "error": "Vision adapter not initialized"}
        source = dsl.get("source")
        name = dsl.get("name")
        
        if not source:
            return {"status": "error", "error": "No source specified"}
        
        # Optional parameters
        kwargs = {}
        for key in ["width", "height", "fps", "rtsp_transport", "use_gstreamer"]:
            if key in dsl:
                kwargs[key] = dsl[key]
        
        success = await self.detector.add_camera(source, name, **kwargs)
        
        if success:
            return {
                "status": "ok",
                "camera": name or source,
                "message": f"Kamera {name or source} dodana"
            }
        else:
            return {
                "status": "error",
                "error": f"Nie można dodać kamery: {source}"
            }
    
    async def _remove_camera(self, dsl: dict) -> dict:
        """Usunięcie kamery."""
        if not self.detector:
            return {"status": "error", "error": "Vision adapter not initialized"}
        name = dsl.get("name") or dsl.get("target")
        
        if not name:
            return {"status": "error", "error": "No camera name specified"}
        
        await self.detector.remove_camera(name)
        
        return {
            "status": "ok",
            "camera": name,
            "message": f"Kamera {name} usunięta"
        }
    
    async def _list_cameras(self, dsl: dict) -> dict:
        """Lista kamer."""
        if not self.detector:
            return {"status": "error", "error": "Vision adapter not initialized"}
        cameras = self.detector.list_cameras()
        
        return {
            "status": "ok",
            "cameras": cameras,
            "count": len(cameras)
        }
    
    async def _detect(self, dsl: dict) -> dict:
        """Wykonanie detekcji."""
        camera = dsl.get("camera")
        
        detections = await self.detector.detect_single(camera_name=camera)
        
        return {
            "status": "ok",
            "detections": [
                {
                    "label": d.label_pl,
                    "confidence": round(d.confidence, 2),
                    "position": d.position,
                    "camera": d.camera
                }
                for d in detections
            ],
            "count": len(detections),
            "description": self.detector.format_detections_for_llm(detections)
        }
    
    async def _describe(self, dsl: dict) -> dict:
        """Opis tego co widać."""
        camera = dsl.get("camera")
        
        detections = await self.detector.detect_single(camera_name=camera)
        
        if not detections:
            description = "Nie widzę żadnych rozpoznawalnych obiektów w kadrze."
        else:
            # Group by label
            counts = {}
            for d in detections:
                counts[d.label_pl] = counts.get(d.label_pl, 0) + 1
            
            parts = []
            for label, count in counts.items():
                if count > 1:
                    parts.append(f"{count}x {label}")
                else:
                    parts.append(label)
            
            description = f"Widzę: {', '.join(parts)}."
            
            # Add positions for first few objects
            if len(detections) <= 5:
                pos_parts = [f"{d.label_pl} ({d.position})" for d in detections]
                description += f" Pozycje: {', '.join(pos_parts)}."
        
        return {
            "status": "ok",
            "description": description,
            "objects": self.detector.format_detections_for_llm(detections)
        }
    
    async def _count_objects(self, dsl: dict) -> dict:
        """Zliczanie obiektów określonego typu."""
        target = dsl.get("target", "").lower()
        camera = dsl.get("camera")
        
        detections = await self.detector.detect_single(camera_name=camera)
        
        if target:
            # Filter by label
            matching = [d for d in detections if target in d.label_pl.lower() or target in d.label.lower()]
            count = len(matching)
            
            return {
                "status": "ok",
                "target": target,
                "count": count,
                "message": f"Widzę {count} obiektów typu '{target}'"
            }
        else:
            return {
                "status": "ok",
                "count": len(detections),
                "message": f"Widzę {len(detections)} obiektów"
            }
    
    async def _find_object(self, dsl: dict) -> dict:
        """Znajdowanie konkretnego obiektu."""
        target = dsl.get("target", "").lower()
        camera = dsl.get("camera")
        
        if not target:
            return {"status": "error", "error": "No target specified"}
        
        detections = await self.detector.detect_single(camera_name=camera)
        
        matching = [d for d in detections if target in d.label_pl.lower() or target in d.label.lower()]
        
        if matching:
            best = max(matching, key=lambda d: d.confidence)
            return {
                "status": "ok",
                "found": True,
                "target": target,
                "position": best.position,
                "confidence": round(best.confidence, 2),
                "message": f"{target.capitalize()} znajduje się {best.position} w kadrze"
            }
        else:
            return {
                "status": "ok",
                "found": False,
                "target": target,
                "message": f"Nie widzę obiektu '{target}' w kadrze"
            }
    
    async def _scan_network(self, dsl: dict) -> dict:
        """Skanowanie sieci w poszukiwaniu kamer RTSP."""
        subnet = dsl.get("subnet", "192.168.1")
        port = dsl.get("port", 554)
        
        self.logger.info(f"Scanning network {subnet}.* for RTSP cameras...")
        
        try:
            cameras = await RTSPDiscovery.scan_network(subnet, port=port)
            
            return {
                "status": "ok",
                "cameras": cameras,
                "count": len(cameras),
                "message": f"Znaleziono {len(cameras)} kamer RTSP"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
