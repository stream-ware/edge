"""
Camera Manager - Unified camera interface

Obsługuje:
- USB cameras (V4L2)
- CSI cameras (Jetson)
- RTSP streams (IP cameras)
- HTTP/MJPEG streams
- File/Video playback
- Multiple cameras simultaneously
"""

import asyncio
import logging
import threading
import queue
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class CameraType(Enum):
    """Typy źródeł wideo."""
    USB = "usb"
    CSI = "csi"
    RTSP = "rtsp"
    HTTP = "http"
    FILE = "file"
    VIRTUAL = "virtual"


@dataclass
class CameraConfig:
    """Konfiguracja kamery."""
    source: Union[int, str]  # Device ID lub URL
    name: str = "default"
    type: CameraType = CameraType.USB
    width: int = 640
    height: int = 480
    fps: int = 30
    
    # RTSP specific
    rtsp_transport: str = "tcp"  # tcp, udp, http
    rtsp_buffer_size: int = 1024000
    
    # Reconnect settings
    reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    
    # GStreamer (Jetson)
    use_gstreamer: bool = False
    use_hw_decode: bool = True  # Hardware decoding on Jetson
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)


class CameraStream:
    """
    Single camera stream handler.
    
    Obsługuje różne źródła wideo z automatycznym reconnect.
    """
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.logger = logging.getLogger(f"camera.{config.name}")
        
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        
        self._frame_queue: queue.Queue = queue.Queue(maxsize=5)
        self._thread: Optional[threading.Thread] = None
        
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0
        self._frame_count: int = 0
        self._fps_actual: float = 0
        
        # Callbacks
        self._on_frame_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []
    
    def _detect_camera_type(self) -> CameraType:
        """Automatyczne wykrycie typu kamery na podstawie source."""
        source = self.config.source
        
        if isinstance(source, int):
            return CameraType.USB
        
        source_str = str(source).lower()
        
        if source_str.startswith("rtsp://"):
            return CameraType.RTSP
        elif source_str.startswith("http://") or source_str.startswith("https://"):
            return CameraType.HTTP
        elif source_str.startswith("csi://") or source_str.startswith("/dev/video"):
            return CameraType.CSI
        elif source_str.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            return CameraType.FILE
        else:
            return CameraType.USB
    
    def _build_pipeline(self) -> str:
        """Budowa pipeline GStreamer dla różnych źródeł."""
        cfg = self.config
        w, h, fps = cfg.width, cfg.height, cfg.fps
        
        # RTSP z hardware decode (Jetson)
        if cfg.type == CameraType.RTSP:
            if cfg.use_hw_decode:
                return (
                    f"rtspsrc location={cfg.source} latency=100 "
                    f"protocols={cfg.rtsp_transport} ! "
                    f"rtph264depay ! h264parse ! nvv4l2decoder ! "
                    f"nvvidconv ! video/x-raw, format=BGRx ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink max-buffers=2 drop=true"
                )
            else:
                return (
                    f"rtspsrc location={cfg.source} latency=100 ! "
                    f"rtph264depay ! avdec_h264 ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink max-buffers=2 drop=true"
                )
        
        # CSI camera (Jetson)
        elif cfg.type == CameraType.CSI:
            sensor_id = 0
            if isinstance(cfg.source, str) and cfg.source.startswith("csi://"):
                sensor_id = int(cfg.source.replace("csi://", ""))
            
            return (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM), width={w}, height={h}, "
                f"framerate={fps}/1, format=NV12 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink max-buffers=2 drop=true"
            )
        
        # USB camera with GStreamer
        elif cfg.type == CameraType.USB:
            device = f"/dev/video{cfg.source}" if isinstance(cfg.source, int) else cfg.source
            return (
                f"v4l2src device={device} ! "
                f"video/x-raw, width={w}, height={h}, framerate={fps}/1 ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink max-buffers=2 drop=true"
            )
        
        # HTTP/MJPEG stream
        elif cfg.type == CameraType.HTTP:
            return (
                f"souphttpsrc location={cfg.source} ! "
                f"multipartdemux ! jpegdec ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink max-buffers=2 drop=true"
            )
        
        return ""
    
    def connect(self) -> bool:
        """Połączenie z kamerą."""
        if cv2 is None:
            self.logger.error("OpenCV not installed")
            return False
        
        # Auto-detect type
        if self.config.type == CameraType.USB and isinstance(self.config.source, str):
            self.config.type = self._detect_camera_type()
        
        self.logger.info(f"Connecting to {self.config.type.value}: {self.config.source}")
        
        try:
            # GStreamer pipeline
            if self.config.use_gstreamer:
                pipeline = self._build_pipeline()
                if pipeline:
                    self.logger.debug(f"GStreamer pipeline: {pipeline}")
                    self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                else:
                    self.cap = cv2.VideoCapture(self.config.source)
            else:
                # Standard OpenCV
                self.cap = cv2.VideoCapture(self.config.source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Cannot open camera: {self.config.source}")
                return False
            
            # Set properties (for non-GStreamer)
            if not self.config.use_gstreamer:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                
                # RTSP buffer
                if self.config.type == CameraType.RTSP:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            self._connected = True
            self._reconnect_count = 0
            
            # Get actual properties
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"✅ Connected: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Rozłączenie."""
        self._connected = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def start(self):
        """Start capture thread."""
        if self._running:
            return
        
        if not self._connected:
            if not self.connect():
                return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.logger.info("Capture thread started")
    
    def stop(self):
        """Stop capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.disconnect()
    
    def _capture_loop(self):
        """Main capture loop (runs in separate thread)."""
        fps_timer = time.time()
        frame_count = 0
        
        while self._running:
            if not self._connected:
                if not self._try_reconnect():
                    break
                continue
            
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.logger.warning("Frame capture failed")
                    self._connected = False
                    continue
                
                self._last_frame = frame
                self._last_frame_time = time.time()
                self._frame_count += 1
                frame_count += 1
                
                # FPS calculation
                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    self._fps_actual = frame_count / elapsed
                    frame_count = 0
                    fps_timer = time.time()
                
                # Put to queue (non-blocking)
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
                    except:
                        pass
                
                # Callbacks
                for callback in self._on_frame_callbacks:
                    try:
                        callback(frame, self.config.name)
                    except Exception as e:
                        self.logger.error(f"Frame callback error: {e}")
                        
            except Exception as e:
                self.logger.error(f"Capture error: {e}")
                self._connected = False
    
    def _try_reconnect(self) -> bool:
        """Próba reconnect."""
        if not self.config.reconnect:
            return False
        
        if self._reconnect_count >= self.config.max_reconnect_attempts:
            self.logger.error("Max reconnect attempts reached")
            self._running = False
            return False
        
        self._reconnect_count += 1
        self.logger.info(f"Reconnecting ({self._reconnect_count}/{self.config.max_reconnect_attempts})...")
        
        time.sleep(self.config.reconnect_delay)
        
        if self.connect():
            return True
        
        return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Pobranie ostatniej klatki (non-blocking)."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return self._last_frame
    
    async def get_frame_async(self) -> Optional[np.ndarray]:
        """Async frame getter."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_frame)
    
    def on_frame(self, callback: Callable):
        """Register frame callback."""
        self._on_frame_callbacks.append(callback)
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def fps(self) -> float:
        return self._fps_actual
    
    @property
    def frame_count(self) -> int:
        return self._frame_count


class CameraManager:
    """
    Manager dla wielu kamer.
    
    Obsługuje:
    - Wiele kamer jednocześnie
    - Dynamiczne dodawanie/usuwanie
    - Unified interface
    """
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("camera_manager")
        self.config = config or {}
        
        self.cameras: Dict[str, CameraStream] = {}
        self._running = False
    
    async def initialize(self):
        """Inicjalizacja z konfiguracji."""
        cameras_config = self.config.get("cameras", [])
        
        for cam_cfg in cameras_config:
            await self.add_camera(
                source=cam_cfg.get("source", 0),
                name=cam_cfg.get("name", f"cam_{len(self.cameras)}"),
                **{k: v for k, v in cam_cfg.items() if k not in ["source", "name"]}
            )
    
    async def add_camera(
        self,
        source: Union[int, str],
        name: str = None,
        camera_type: str = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        **kwargs
    ) -> Optional[CameraStream]:
        """
        Dodanie kamery.
        
        Args:
            source: Device ID, URL RTSP, HTTP, lub ścieżka do pliku
            name: Nazwa kamery (unikalna)
            camera_type: Typ (usb, csi, rtsp, http, file) lub auto-detect
            width, height, fps: Parametry wideo
            **kwargs: Dodatkowe opcje (rtsp_transport, use_gstreamer, etc.)
        
        Examples:
            # USB camera
            await manager.add_camera(0, name="usb_cam")
            
            # RTSP stream
            await manager.add_camera(
                "rtsp://192.168.1.100:554/stream1",
                name="ip_camera_1",
                rtsp_transport="tcp"
            )
            
            # HTTP MJPEG
            await manager.add_camera(
                "http://192.168.1.101/video.mjpg",
                name="webcam"
            )
        """
        if name is None:
            name = f"camera_{len(self.cameras)}"
        
        if name in self.cameras:
            self.logger.warning(f"Camera {name} already exists")
            return self.cameras[name]
        
        # Determine camera type
        if camera_type:
            cam_type = CameraType(camera_type.lower())
        else:
            cam_type = CameraType.USB  # Will be auto-detected
        
        config = CameraConfig(
            source=source,
            name=name,
            type=cam_type,
            width=width,
            height=height,
            fps=fps,
            **kwargs
        )
        
        camera = CameraStream(config)
        
        # Connect and start
        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, camera.connect)
        
        if connected:
            camera.start()
            self.cameras[name] = camera
            self.logger.info(f"✅ Added camera: {name}")
            return camera
        else:
            self.logger.error(f"Failed to add camera: {name}")
            return None
    
    async def remove_camera(self, name: str):
        """Usunięcie kamery."""
        if name in self.cameras:
            camera = self.cameras.pop(name)
            camera.stop()
            self.logger.info(f"Removed camera: {name}")
    
    def get_camera(self, name: str) -> Optional[CameraStream]:
        """Pobranie kamery po nazwie."""
        return self.cameras.get(name)
    
    async def get_frame(self, name: str = None) -> Optional[np.ndarray]:
        """
        Pobranie klatki z kamery.
        
        Args:
            name: Nazwa kamery (None = pierwsza dostępna)
        """
        if name:
            camera = self.cameras.get(name)
        else:
            # First available camera
            camera = next(iter(self.cameras.values()), None)
        
        if camera:
            return await camera.get_frame_async()
        return None
    
    async def get_all_frames(self) -> Dict[str, np.ndarray]:
        """Pobranie klatek ze wszystkich kamer."""
        frames = {}
        for name, camera in self.cameras.items():
            frame = await camera.get_frame_async()
            if frame is not None:
                frames[name] = frame
        return frames
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """Lista wszystkich kamer z ich statusem."""
        result = []
        for name, camera in self.cameras.items():
            result.append({
                "name": name,
                "source": camera.config.source,
                "type": camera.config.type.value,
                "connected": camera.is_connected,
                "fps": round(camera.fps, 1),
                "frames": camera.frame_count
            })
        return result
    
    async def cleanup(self):
        """Zamknięcie wszystkich kamer."""
        for name in list(self.cameras.keys()):
            await self.remove_camera(name)


# ============================================
# RTSP Discovery (optional)
# ============================================

class RTSPDiscovery:
    """Odkrywanie kamer RTSP w sieci lokalnej."""
    
    # Popularne ścieżki RTSP dla różnych producentów
    COMMON_PATHS = [
        "/",
        "/stream1",
        "/stream/main",
        "/live/ch0",
        "/live/main",
        "/h264/ch1/main/av_stream",
        "/cam/realmonitor?channel=1&subtype=0",
        "/Streaming/Channels/101",
        "/video1",
        "/videoMain",
    ]
    
    # Domyślne porty RTSP
    COMMON_PORTS = [554, 8554, 5554]
    
    @classmethod
    async def probe_camera(
        cls, 
        host: str, 
        port: int = 554,
        username: str = None,
        password: str = None,
        timeout: float = 5.0
    ) -> Optional[str]:
        """
        Próba znalezienia działającego URL RTSP.
        
        Returns:
            Działający URL RTSP lub None
        """
        if cv2 is None:
            return None
        
        auth = ""
        if username and password:
            auth = f"{username}:{password}@"
        
        for path in cls.COMMON_PATHS:
            url = f"rtsp://{auth}{host}:{port}{path}"
            
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
                
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        return url
                
                cap.release()
            except:
                pass
        
        return None
    
    @classmethod
    async def scan_network(
        cls,
        subnet: str = "192.168.1",
        start: int = 1,
        end: int = 254,
        port: int = 554
    ) -> List[str]:
        """
        Skanowanie sieci w poszukiwaniu kamer RTSP.
        
        Returns:
            Lista znalezionych URL RTSP
        """
        import socket
        
        found = []
        
        async def check_host(ip: str):
            # Quick TCP check first
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((ip, port))
            sock.close()
            
            if result == 0:
                url = await cls.probe_camera(ip, port, timeout=3.0)
                if url:
                    found.append(url)
        
        tasks = []
        for i in range(start, end + 1):
            ip = f"{subnet}.{i}"
            tasks.append(asyncio.create_task(check_host(ip)))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return found
