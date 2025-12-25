# 🎯 Streamware Orchestrator

**LLM-powered Docker/IoT Orchestrator z interfejsem głosowym i wizyjnym**

Integracja:
- **Audio Interface** (STT/TTS) - Faster-Whisper + Piper
- **Vision Interface** - YOLOv8 + Multi-camera support (USB/RTSP/HTTP)
- **LLM Orchestrator** - Ollama/Phi-3
- **Text2DSL** - Natural Language → Domain Specific Language
- **MQTT** - Komunikacja z urządzeniami IoT/Edge
- **Docker Control** - Zarządzanie kontenerami głosem

## 🏗️ Architektura

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMWARE ORCHESTRATOR                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Mikrofon] ──► [STT/Whisper] ──┐                               │
│                                  ├──► [LLM/Ollama]              │
│  [Kamery] ───► [Vision/YOLO] ───┘         │                     │
│   ├─ USB                             [Text2DSL]                  │
│   ├─ RTSP (IP)                            │                     │
│   └─ HTTP/MJPEG          ┌────────────────┼────────────────┐    │
│                          │                │                │    │
│                    [Docker]         [Vision]         [MQTT]     │
│                    Adapter          Adapter         Adapter     │
│                          │                │                │    │
│                          └────────────────┼────────────────┘    │
│                                           │                     │
│                                      [MQTT Broker]              │
│                                           │                     │
│  [Głośnik] ◄── [TTS/Piper] ◄─────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

## 📹 Obsługiwane źródła wideo

| Typ | Przykład | Opis |
|-----|----------|------|
| **USB** | `0`, `1`, `/dev/video0` | Kamera USB/V4L2 |
| **CSI** | `csi://0` | Kamera CSI (Jetson) |
| **RTSP** | `rtsp://192.168.1.100:554/stream` | Kamery IP |
| **HTTP** | `http://192.168.1.101/video.mjpg` | Streamy MJPEG |
| **File** | `/path/to/video.mp4` | Pliki wideo |

## 📋 Komendy głosowe

### Docker

| Komenda | Akcja DSL |
|---------|-----------|
| "Zrestartuj backend" | `docker.restart` |
| "Pokaż logi frontendu" | `docker.logs` |
| "Status kontenerów" | `docker.status` |

### Vision / Kamera

| Komenda | Akcja DSL |
|---------|-----------|
| "Co widzisz?" | `vision.describe` |
| "Ile osób widzisz?" | `vision.count` |
| "Gdzie jest kubek?" | `vision.find` |
| "Dodaj kamerę rtsp://..." | `vision.add_camera` |
| "Lista kamer" | `vision.list_cameras` |
| "Skanuj sieć RTSP" | `vision.scan_network` |

### IoT / Sensory

| Komenda | Akcja DSL |
|---------|-----------|
| "Jaka jest temperatura?" | `sensor.read` |
| "Włącz światło w kuchni" | `device.set` |

### Text2DSL - przykłady transformacji

```
Natural Language                    →  DSL (JSON)
────────────────────────────────────────────────────────────────
"Zrestartuj backend"               →  {"action": "docker.restart", 
                                        "target": "backend"}

"Pokaż ostatnie 20 linii logów"    →  {"action": "docker.logs",
                                        "target": "backend", 
                                        "tail": 20}

"Jaka jest temperatura w salonie?" →  {"action": "sensor.read",
                                        "device": "salon",
                                        "metric": "temperature"}
```

## 🚀 Wdrożenia Docker

### Deployment 1: Single Container (Development)

```bash
docker-compose -f docker-compose-single.yml up
```

### Deployment 2: Multi-Service (Staging)

```bash
docker-compose -f docker-compose-multi.yml up
```

### Deployment 3: Full Edge + Backend (Production)

```bash
docker-compose -f docker-compose-full.yml up
```

## 📁 Struktura projektu

```
streamware-orchestrator/
├── orchestrator/
│   ├── main.py                 # Entry point z MQTT + Audio
│   ├── text2dsl.py             # Konwersja NL ↔ DSL
│   ├── llm_engine.py           # LLM wrapper (Ollama)
│   ├── audio/
│   │   ├── stt.py              # Speech-to-Text (Whisper)
│   │   └── tts.py              # Text-to-Speech (Piper)
│   ├── adapters/
│   │   ├── docker_adapter.py   # Docker API
│   │   ├── sql_adapter.py      # PostgreSQL
│   │   ├── mqtt_adapter.py     # MQTT client
│   │   └── firmware_adapter.py # IoT devices
│   ├── Dockerfile
│   └── requirements.txt
│
├── firmware/
│   └── sim.py                  # Symulator czujników IoT
│
├── docker-compose-single.yml   # Dev deployment
├── docker-compose-multi.yml    # Staging deployment
├── docker-compose-full.yml     # Production deployment
│
├── config/
│   ├── config.yaml             # Główna konfiguracja
│   └── mosquitto.conf          # MQTT broker config
│
└── models/                     # Modele AI (pobierane)
    ├── whisper/
    └── piper/
```

## ⚙️ Instalacja

### Lokalna (bez Docker)

```bash
# Klonuj repo
git clone https://github.com/softreck/streamware-orchestrator.git
cd streamware-orchestrator

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Zależności
pip install -r orchestrator/requirements.txt

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# Uruchom
python orchestrator/main.py
```

### Docker (zalecane)

```bash
docker-compose -f docker-compose-full.yml up --build
```

## 🎤 Użycie

Po uruchomieniu system nasłuchuje na mikrofonie.

**Przykładowa sesja:**

```
🎤 Nasłuchuję...

Ty: "Pokaż status kontenerów"

🤖 Orchestrator:
   DSL: {"action": "docker.status"}
   Wykonuję...
   
🔊 "Masz uruchomione 4 kontenery: backend, frontend, 
    database i mqtt broker. Wszystkie działają poprawnie."

Ty: "Zrestartuj backend"

🤖 Orchestrator:
   DSL: {"action": "docker.restart", "target": "backend"}
   Wykonuję...

🔊 "Kontener backend został zrestartowany pomyślnie."
```

## 🔧 Konfiguracja

```yaml
# config/config.yaml
audio:
  stt:
    model: "small"
    language: "pl"
  tts:
    model: "pl_PL-gosia-medium"
    
llm:
  provider: "ollama"
  model: "phi3:mini"
  
mqtt:
  broker: "localhost"
  port: 1883
  topics:
    commands: "commands/#"
    events: "events/#"
    sensors: "edge/sensors"

docker:
  socket: "unix:///var/run/docker.sock"
  
adapters:
  enabled:
    - docker
    - mqtt
    - sql
```

## 📡 MQTT Topics

| Topic | Kierunek | Opis |
|-------|----------|------|
| `commands/{target}` | IN | Komendy do wykonania |
| `events/{target}` | OUT | Wyniki akcji |
| `edge/sensors` | IN | Dane z czujników IoT |
| `audio/stt` | OUT | Rozpoznany tekst |
| `audio/tts` | IN | Tekst do wymówienia |

## 🔌 Rozszerzanie

### Własny adapter

```python
# orchestrator/adapters/my_adapter.py
from .base import BaseAdapter

class MyAdapter(BaseAdapter):
    name = "myservice"
    
    def execute(self, dsl: dict) -> dict:
        action = dsl.get("action")
        
        if action == "myservice.hello":
            return {"status": "ok", "message": "Hello!"}
        
        return {"status": "error", "message": "Unknown action"}
```

### Własne komendy DSL

```python
# orchestrator/text2dsl.py - dodaj pattern
PATTERNS = {
    ...
    r"przywitaj się": {"action": "myservice.hello"},
}
```

## 📊 Wydajność

| Komponent | Latency | RAM |
|-----------|---------|-----|
| STT (Whisper small) | ~200ms | ~500MB |
| LLM (Phi-3 Mini) | ~300ms | ~4GB |
| Text2DSL | <10ms | ~10MB |
| Docker API | ~50ms | ~20MB |
| TTS (Piper) | ~100ms | ~200MB |
| **TOTAL** | **~700ms** | **~5GB** |

## 📄 Licencja

MIT License - Softreck / prototypowanie.pl
