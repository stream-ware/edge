# 📹 Camera Configuration Guide

## Quick Start

### USB Camera (najprostsza konfiguracja)

```yaml
# config/config.yaml
vision:
  cameras:
    - source: 0
      name: "laptop_cam"
```

### RTSP Camera (kamera IP)

```yaml
vision:
  cameras:
    - source: "rtsp://192.168.1.100:554/stream1"
      name: "ip_camera"
      rtsp_transport: "tcp"  # tcp, udp, http
      use_gstreamer: true    # Zalecane dla RTSP
      use_hw_decode: true    # Hardware decode (Jetson)
```

### HTTP MJPEG Stream

```yaml
vision:
  cameras:
    - source: "http://192.168.1.101:8080/video.mjpg"
      name: "webcam"
```

### Multiple Cameras

```yaml
vision:
  cameras:
    - source: 0
      name: "usb_front"
      
    - source: "rtsp://192.168.1.100:554/stream"
      name: "entrance"
      rtsp_transport: "tcp"
      
    - source: "rtsp://192.168.1.101:554/stream"
      name: "parking"
      width: 1920
      height: 1080
```

---

## Komendy głosowe

### Podstawowe

```
"Co widzisz?"
→ Opisuje wszystkie wykryte obiekty

"Co widzisz przez kamerę entrance?"
→ Opisuje obiekty z konkretnej kamery

"Ile osób widzisz?"
→ Zlicza osoby

"Gdzie jest kubek?"
→ Znajduje pozycję obiektu
```

### Zarządzanie kamerami

```
"Dodaj kamerę rtsp://192.168.1.100:554/stream"
→ Dodaje nową kamerę RTSP

"Dodaj kamerę 0"
→ Dodaje kamerę USB

"Usuń kamerę parking"
→ Odłącza kamerę

"Lista kamer"
→ Pokazuje wszystkie kamery i ich status

"Skanuj sieć RTSP"
→ Szuka kamer RTSP w sieci lokalnej
```

---

## Programmatic API

### Python

```python
from vision import VisionAdapter

# Inicjalizacja
vision = VisionAdapter(config)
await vision.initialize()

# Dodaj kamerę USB
await vision.execute({
    "action": "vision.add_camera",
    "source": 0,
    "name": "usb_cam"
})

# Dodaj kamerę RTSP
await vision.execute({
    "action": "vision.add_camera",
    "source": "rtsp://192.168.1.100:554/stream",
    "name": "ip_camera",
    "rtsp_transport": "tcp"
})

# Wykonaj detekcję
result = await vision.execute({
    "action": "vision.detect"
})
print(result["detections"])

# Opisz scenę
result = await vision.execute({
    "action": "vision.describe"
})
print(result["description"])

# Znajdź obiekt
result = await vision.execute({
    "action": "vision.find",
    "target": "person"
})
if result["found"]:
    print(f"Osoba: {result['position']}")
```

### MQTT

```bash
# Komenda przez MQTT
mosquitto_pub -t "commands/vision" -m "co widzisz"

# Dodaj kamerę
mosquitto_pub -t "commands/vision" -m '{"action":"vision.add_camera","source":"rtsp://192.168.1.100:554/stream"}'

# Subskrybuj wyniki
mosquitto_sub -t "events/vision"
```

---

## Producenci kamer RTSP

### Popularne ścieżki RTSP

| Producent | Typowa ścieżka |
|-----------|----------------|
| **Hikvision** | `/Streaming/Channels/101` |
| **Dahua** | `/cam/realmonitor?channel=1&subtype=0` |
| **Axis** | `/axis-media/media.amp` |
| **Reolink** | `/h264Preview_01_main` |
| **Amcrest** | `/cam/realmonitor?channel=1&subtype=0` |
| **Foscam** | `/videoMain` |
| **Generic** | `/stream1`, `/live/ch0`, `/` |

### Przykłady pełnych URL

```
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# Reolink
rtsp://admin:password@192.168.1.100:554/h264Preview_01_main

# Generic (bez auth)
rtsp://192.168.1.100:554/stream1
```

---

## Optymalizacja dla Jetson

### GStreamer + Hardware Decode

```yaml
vision:
  cameras:
    - source: "rtsp://192.168.1.100:554/stream"
      name: "ip_cam"
      use_gstreamer: true
      use_hw_decode: true  # nvv4l2decoder
      width: 1280
      height: 720
```

### TensorRT dla YOLO

```bash
# Export modelu do TensorRT
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True)
"

# Użyj w konfiguracji
vision:
  model_path: "models/yolov8n.engine"
```

### Zmniejszenie obciążenia

```yaml
vision:
  process_every_n_frames: 10  # Przetwarzaj co 10 klatek
  confidence: 0.6             # Wyższa pewność = mniej fałszywych
  max_detections: 10          # Limit detekcji
  
  cameras:
    - source: "rtsp://..."
      width: 640    # Niższa rozdzielczość
      height: 480
      fps: 15       # Niższy FPS
```

---

## Troubleshooting

### RTSP nie łączy się

```bash
# Test z ffmpeg
ffmpeg -i "rtsp://192.168.1.100:554/stream" -frames 1 test.jpg

# Sprawdź firewall
sudo ufw allow 554/tcp

# Spróbuj różnych transportów
rtsp_transport: "udp"  # zamiast "tcp"
```

### Wysokie opóźnienie RTSP

```yaml
# Zmniejsz bufor
cameras:
  - source: "rtsp://..."
    rtsp_buffer_size: 512000  # Mniejszy bufor
```

### Brak kamery USB

```bash
# Sprawdź urządzenia
ls -la /dev/video*
v4l2-ctl --list-devices

# Uprawnienia
sudo usermod -aG video $USER
```

### GStreamer nie działa

```bash
# Instalacja na Ubuntu
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-bad

# Test pipeline
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```
