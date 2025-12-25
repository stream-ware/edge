# 📡 API Reference

## 1. DSL Actions

### 1.1 Docker Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `docker.restart` | `target` | Restart kontenera |
| `docker.stop` | `target` | Zatrzymanie kontenera |
| `docker.start` | `target` | Uruchomienie kontenera |
| `docker.logs` | `target`, `tail` | Pobranie logów |
| `docker.status` | - | Status wszystkich kontenerów |
| `docker.inspect` | `target` | Szczegóły kontenera |
| `docker.list` | - | Lista kontenerów |

**Przykłady DSL:**

```json
{"action": "docker.restart", "target": "backend"}
{"action": "docker.logs", "target": "frontend", "tail": 20}
{"action": "docker.status"}
```

### 1.2 Vision Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `vision.describe` | `camera?` | Opis sceny |
| `vision.detect` | `camera?` | Lista detekcji |
| `vision.count` | `target`, `camera?` | Zliczanie obiektów |
| `vision.find` | `target`, `camera?` | Znajdowanie obiektu |
| `vision.add_camera` | `source`, `name?`, ... | Dodanie kamery |
| `vision.remove_camera` | `name` | Usunięcie kamery |
| `vision.list_cameras` | - | Lista kamer |
| `vision.scan_network` | `subnet?`, `port?` | Skanowanie RTSP |

**Przykłady DSL:**

```json
{"action": "vision.describe"}
{"action": "vision.count", "target": "person"}
{"action": "vision.find", "target": "laptop"}
{"action": "vision.add_camera", "source": "rtsp://192.168.1.100:554/stream", "name": "ip_cam"}
```

### 1.3 Sensor Actions

| Action | Parameters | Description |
|--------|------------|-------------|
| `sensor.read` | `metric`, `location?` | Odczyt sensora |
| `device.set` | `device`, `location?`, `state`/`value` | Ustawienie urządzenia |
| `device.get` | `location` | Stan urządzenia |

**Przykłady DSL:**

```json
{"action": "sensor.read", "metric": "temperature", "location": "salon"}
{"action": "device.set", "device": "light", "location": "kuchnia", "state": "on"}
```

### 1.4 System Actions

| Action | Description |
|--------|-------------|
| `system.help` | Wyświetla pomoc |
| `system.exit` | Kończy sesję |
| `system.mute` | Wycisza TTS |

---

## 2. MQTT Topics

### 2.1 Input Topics

| Topic | Payload | Description |
|-------|---------|-------------|
| `commands/#` | Natural language lub JSON | Komendy do wykonania |
| `commands/{target}` | NL/JSON | Komenda z target hint |
| `audio/tts` | string | Tekst do wypowiedzenia |
| `edge/sensors` | JSON | Dane z sensorów |

**Przykłady:**

```bash
# Komenda tekstowa
mosquitto_pub -t "commands/backend" -m "pokaż logi"

# Komenda JSON
mosquitto_pub -t "commands/vision" -m '{"action":"vision.describe"}'

# TTS
mosquitto_pub -t "audio/tts" -m "Witaj w systemie"
```

### 2.2 Output Topics

| Topic | Payload | Description |
|-------|---------|-------------|
| `events/{target}` | JSON | Wynik wykonania akcji |
| `audio/stt` | string | Rozpoznany tekst |

**Przykład eventu:**

```json
{
  "response": "Kontener backend został zrestartowany",
  "dsl": {
    "action": "docker.restart",
    "target": "backend",
    "status": "ok"
  },
  "source": "audio"
}
```

---

## 3. Python API

### 3.1 Orchestrator

```python
from orchestrator import Orchestrator

# Inicjalizacja
orch = Orchestrator("config/config.yaml")
await orch.initialize()

# Przetwarzanie komendy
response = await orch.process_command(
    text="Zrestartuj backend",
    source="api"
)
print(response)  # "Kontener backend został zrestartowany"

# Zatrzymanie
await orch.stop()
await orch.cleanup()
```

### 3.2 Text2DSL

```python
from text2dsl import Text2DSL

converter = Text2DSL()

# NL → DSL
dsl = converter.nl_to_dsl("pokaż logi frontend 20 linii")
# {"action": "docker.logs", "target": "frontend", "tail": 20}

# DSL → NL
result = {"action": "docker.logs", "target": "frontend", "logs": "..."}
response = converter.dsl_to_nl(result)
# "Ostatnie logi z frontend: ..."

# LLM fallback prompt
prompt = converter.get_llm_prompt("zrób coś dziwnego")
```

### 3.3 Vision Adapter

```python
from vision import VisionAdapter

vision = VisionAdapter(config)
await vision.initialize()

# Dodaj kamerę
result = await vision.execute({
    "action": "vision.add_camera",
    "source": "rtsp://192.168.1.100:554/stream",
    "name": "ip_camera",
    "rtsp_transport": "tcp"
})

# Opisz scenę
result = await vision.execute({"action": "vision.describe"})
print(result["description"])

# Znajdź obiekt
result = await vision.execute({
    "action": "vision.find",
    "target": "person"
})
if result["found"]:
    print(f"Pozycja: {result['position']}")

# Lista kamer
cameras = await vision.execute({"action": "vision.list_cameras"})
```

### 3.4 Camera Manager

```python
from vision import CameraManager, CameraConfig, CameraType

manager = CameraManager()

# Dodaj USB camera
await manager.add_camera(0, name="usb_cam")

# Dodaj RTSP camera
await manager.add_camera(
    "rtsp://192.168.1.100:554/stream",
    name="ip_cam",
    rtsp_transport="tcp",
    use_gstreamer=True
)

# Pobierz klatkę
frame = await manager.get_frame("ip_cam")

# Wszystkie klatki
frames = await manager.get_all_frames()

# Lista kamer
for cam in manager.list_cameras():
    print(f"{cam['name']}: {cam['connected']}")

await manager.cleanup()
```

### 3.5 Object Detector

```python
from vision import ObjectDetector

detector = ObjectDetector({
    "model": "yolov8n",
    "confidence": 0.5
})
await detector.initialize()

# Dodaj kamery
await detector.add_camera(0, name="cam1")
await detector.add_camera("rtsp://...", name="cam2")

# Stream detekcji
async for detections in detector.stream():
    for d in detections:
        print(f"{d.label_pl}: {d.position} ({d.confidence:.0%})")

# Pojedyncza detekcja
detections = await detector.detect_single(camera_name="cam1")

# Format dla LLM
context = detector.format_detections_for_llm()
# "[osoba: lewo, 92%], [kubek: środek, 87%]"
```

### 3.6 Audio STT

```python
from audio import SpeechToText

stt = SpeechToText(
    stt_config={"model": "small", "language": "pl"},
    audio_config={"sample_rate": 16000}
)
await stt.initialize()

# Stream transkrypcji
async for transcript in stt.stream():
    print(f"Recognized: {transcript}")

await stt.cleanup()
```

### 3.7 Audio TTS

```python
from audio import TextToSpeech

tts = TextToSpeech({"model": "pl_PL-gosia-medium"})
await tts.initialize()

# Mów
await tts.speak("Witaj w systemie Streamware")

# Do pliku
await tts.synthesize_to_file("Tekst", "output.wav")

await tts.cleanup()
```

---

## 4. Response Format

### 4.1 Success Response

```json
{
  "status": "ok",
  "action": "docker.restart",
  "target": "backend",
  "message": "Kontener backend został zrestartowany"
}
```

### 4.2 Error Response

```json
{
  "status": "error",
  "action": "docker.restart",
  "target": "nonexistent",
  "error": "Container 'nonexistent' not found"
}
```

### 4.3 Detection Response

```json
{
  "status": "ok",
  "action": "vision.detect",
  "detections": [
    {
      "label": "osoba",
      "confidence": 0.92,
      "position": "lewo",
      "camera": "usb_cam"
    },
    {
      "label": "kubek",
      "confidence": 0.87,
      "position": "środek",
      "camera": "usb_cam"
    }
  ],
  "count": 2,
  "description": "[osoba: lewo, 92%], [kubek: środek, 87%]"
}
```

---

## 5. Natural Language Patterns

### 5.1 Polish Commands

```
# Docker
"zrestartuj {container}"
"zatrzymaj {container}"
"uruchom {container}"
"pokaż logi {container}"
"logi {container} {N} linii"
"status kontenerów"

# Vision
"co widzisz"
"opisz co widzisz"
"ile {object} widzisz"
"gdzie jest {object}"
"znajdź {object}"
"dodaj kamerę {source}"
"usuń kamerę {name}"
"lista kamer"
"skanuj sieć"

# Sensors
"jaka jest temperatura"
"podaj wilgotność w {location}"
"włącz światło w {location}"
"wyłącz {device}"
"ustaw {device} na {value}"

# System
"pomoc"
"koniec"
"cicho"
```

### 5.2 English Commands

```
# Docker
"restart {container}"
"stop {container}"
"start {container}"
"show logs {container}"
"container status"

# Vision
"what do you see"
"describe the scene"
"how many {object}"
"where is {object}"
"find {object}"
"add camera {source}"
"list cameras"
"scan network"

# Sensors
"what is the temperature"
"turn on the light"
"turn off {device}"
```
