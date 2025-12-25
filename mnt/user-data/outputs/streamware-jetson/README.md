# 🤖 Streamware Jetson - Lokalny Asystent Wizyjno-Głosowy

**Real-time audio/video AI assistant dla NVIDIA Jetson Orin Nano 8GB**

## 🎯 Funkcjonalności

- **Speech-to-Text**: Rozpoznawanie mowy w czasie rzeczywistym (PL/EN)
- **Vision AI**: Detekcja obiektów przez kamerę
- **LLM**: Lokalne przetwarzanie języka naturalnego
- **Text-to-Speech**: Synteza mowy w języku polskim
- **Zero nagrywania**: Wszystko w RAM, zgodność z RODO

## 📊 Architektura

```
┌─────────────┐     ┌─────────────┐
│  Mikrofon   │────►│  STT        │
│  (PyAudio)  │     │  (Whisper)  │
└─────────────┘     └──────┬──────┘
                           │
┌─────────────┐            │      ┌─────────────┐
│  Kamera     │────►┌──────▼──────┤  Orchestrator│────►│  TTS        │
│  (OpenCV)   │     │             │  (Asyncio)   │     │  (Piper)    │
└─────────────┘     │             └──────┬───────┘     └─────────────┘
       │            │                    │
       ▼            │                    ▼
┌─────────────┐     │             ┌─────────────┐
│  Vision     │─────┘             │  LLM        │
│  (YOLOv8)   │                   │  (Ollama)   │
└─────────────┘                   └─────────────┘
```

## 🔧 Stack technologiczny

| Komponent | Technologia | Uzasadnienie |
|-----------|-------------|--------------|
| STT | **Faster-Whisper small** | Optymalny balans prędkość/jakość na GPU |
| Vision | **YOLOv8n + TensorRT** | Natywna akceleracja Jetson |
| LLM | **Ollama + Phi-3 Mini** | 3.8B parametrów, mieści się w 8GB |
| TTS | **Piper TTS** | Ultra-lekki, dobra jakość PL |
| Audio I/O | **PyAudio + sounddevice** | Niskie latency |
| Video I/O | **OpenCV + GStreamer** | Hardware decode na Jetson |
| IPC | **asyncio + queues** | Zero overhead, single process |

## 📋 Wymagania

### Hardware
- NVIDIA Jetson Orin Nano 8GB
- Mikrofon USB (lub I2S)
- Kamera USB/CSI
- Głośnik/słuchawki

### Software
- JetPack 6.0+ (Ubuntu 22.04)
- CUDA 12.2+
- Python 3.10+

## 🚀 Instalacja

### 1. Przygotowanie systemu

```bash
# Aktualizacja
sudo apt update && sudo apt upgrade -y

# Podstawowe zależności
sudo apt install -y \
    python3-pip python3-venv \
    portaudio19-dev libsndfile1 \
    libopencv-dev ffmpeg \
    espeak-ng libespeak-ng-dev
```

### 2. Klonowanie i setup

```bash
git clone https://github.com/softreck/streamware-jetson.git
cd streamware-jetson

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Instalacja zależności
pip install -r requirements.txt
```

### 3. Modele

```bash
# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# Whisper
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cuda')"

# Piper TTS (polski głos)
./scripts/download_piper_pl.sh

# YOLOv8 TensorRT
python scripts/export_yolo_tensorrt.py
```

### 4. Uruchomienie

```bash
python main.py
```

## 📁 Struktura projektu

```
streamware-jetson/
├── main.py                 # Entry point
├── requirements.txt        # Zależności Python
├── config.yaml            # Konfiguracja
│
├── src/
│   ├── __init__.py
│   ├── orchestrator.py    # Główna logika
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── stt.py         # Speech-to-Text
│   │   └── tts.py         # Text-to-Speech
│   ├── vision/
│   │   ├── __init__.py
│   │   └── detector.py    # Detekcja obiektów
│   └── llm/
│       ├── __init__.py
│       └── inference.py   # LLM wrapper
│
├── models/
│   ├── whisper/           # Faster-Whisper
│   ├── yolo/              # YOLOv8 TensorRT
│   └── piper/             # Piper TTS
│
├── scripts/
│   ├── download_piper_pl.sh
│   ├── export_yolo_tensorrt.py
│   └── benchmark.py
│
└── tests/
    ├── test_stt.py
    ├── test_vision.py
    └── test_tts.py
```

## ⚙️ Konfiguracja

```yaml
# config.yaml
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  vad_threshold: 0.5

stt:
  model: "small"
  language: "pl"
  beam_size: 5
  compute_type: "float16"

vision:
  model: "yolov8n"
  confidence: 0.5
  process_every_n_frames: 5
  resolution: [640, 480]

llm:
  model: "phi3:mini"
  temperature: 0.7
  max_tokens: 256
  system_prompt: |
    Jesteś pomocnym asystentem wizyjno-głosowym.
    Odpowiadasz krótko i konkretnie po polsku.
    Masz dostęp do informacji o obiektach widzianych przez kamerę.

tts:
  model: "pl_PL-gosia-medium"
  speaker_id: 0
  length_scale: 1.0
```

## 🎮 Użycie

### Podstawowe komendy głosowe

| Komenda | Działanie |
|---------|-----------|
| "Co widzisz?" | Opis obiektów w polu widzenia |
| "Ile jest [obiektów]?" | Zliczanie obiektów danego typu |
| "Gdzie jest [obiekt]?" | Lokalizacja obiektu w kadrze |
| "Opisz scenę" | Pełny opis widzianej sceny |
| "Stop" / "Koniec" | Zakończenie sesji |

### API (opcjonalne)

```python
from streamware import Assistant

assistant = Assistant(config="config.yaml")
assistant.start()

# Programowe zapytanie
response = assistant.query(
    text="Co leży na stole?",
    include_vision=True
)
print(response)
```

## 📈 Wydajność

| Metryka | Wartość |
|---------|---------|
| Latency STT | ~200ms |
| Latency Vision | ~50ms (co 5 klatek) |
| Latency LLM | ~300-500ms |
| **Total latency** | **~600-900ms** |
| RAM usage | ~5-6GB |
| GPU usage | ~70-80% |

## 🔌 Rozszerzenia

### Dodanie bufora (z nagrywaniem)

```python
# config.yaml
buffer:
  enabled: true
  audio_seconds: 30
  video_frames: 150  # 5s @ 30fps
```

### Integracja z Home Assistant

```yaml
# home_assistant.yaml
homeassistant:
  enabled: true
  url: "http://192.168.1.100:8123"
  token: "${HA_TOKEN}"
```

### WebSocket API

```yaml
api:
  enabled: true
  host: "0.0.0.0"
  port: 8765
```

## 🐛 Troubleshooting

### Problem: Brak dźwięku z mikrofonu

```bash
# Sprawdź urządzenia
arecord -l
# Ustaw domyślne
export AUDIODEV=hw:1,0
```

### Problem: CUDA out of memory

```bash
# Zmniejsz model whisper
stt:
  model: "tiny"  # zamiast "small"
```

### Problem: Niska jakość TTS

```bash
# Użyj lepszego głosu
./scripts/download_piper_pl.sh --quality high
```

## 📄 Licencja

MIT License - używaj dowolnie w projektach komercyjnych i niekomercyjnych.

## 🤝 Współpraca

Projekt rozwijany przez [Softreck](https://softreck.com) w ramach [prototypowanie.pl](https://prototypowanie.pl).

Issues i PR-y mile widziane!
