# 🎯 Streamware - Kompletny System AI Audio/Video

**Zunifikowane rozwiązanie AI z obsługą głosu, wizji i automatyzacji**

Jeden projekt obsługujący dwa tryby wdrożenia:
- **Docker/Enterprise** - dla serwerów i środowisk produkcyjnych
- **Embedded/Standalone** - dla NVIDIA Jetson i urządzeń edge

---

## 📦 Struktura projektu

```
streamware/
├── orchestrator/           # Główny kod aplikacji
│   ├── main.py            # Entry point
│   ├── text2dsl.py        # NL ↔ DSL conversion
│   ├── llm_engine.py      # LLM wrapper
│   ├── audio/             # STT + TTS
│   ├── vision/            # Cameras + Detection
│   └── adapters/          # Docker, MQTT, Firmware
├── config/                # Konfiguracja
│   └── config.yaml        # Unified config (mode: docker/embedded)
├── scripts/               # Skrypty instalacyjne i pomocnicze
├── tests/                 # Testy jednostkowe
├── docs/                  # Dokumentacja
├── firmware/              # IoT simulator
├── models/                # Modele AI (whisper, yolo, piper)
├── docker-compose-*.yml   # Docker deployment configs
├── Makefile              # Build & run commands
└── requirements.txt       # Python dependencies
```

---

## 🏗️ Architektura

```
┌─────────────────────────────────────────────────────────────┐
│                      STREAMWARE                              │
├─────────────────────────────────────────────────────────────┤
│  [Audio STT] ──┐                                            │
│   (Whisper)    ├──► [LLM] ──► [Text2DSL] ──► [Adapters]    │
│  [Cameras]  ───┘   (Ollama)                      │          │
│   ├─ USB                                    ┌────┴────┐     │
│   ├─ RTSP (IP)                             │         │     │
│   └─ CSI (Jetson)     [Vision]          [Docker] [MQTT]    │
│                       (YOLO)               │         │     │
│  [TTS Piper] ◄────────────────────   Containers  IoT       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Szybki start

### Tryb Docker (Enterprise)

```bash
# Konfiguracja
cp config/config.yaml config/config.yaml.local
# Ustaw: mode: "docker"

# Development (pojedynczy kontener)
make docker-dev

# Production (pełny stack)
make docker-prod

# Lub bezpośrednio:
docker-compose -f docker-compose-single.yml up
```

### Tryb Embedded (Jetson/Edge)

```bash
# Instalacja
make install

# Lub ręcznie:
./scripts/install.sh
source venv/bin/activate

# Konfiguracja
# Ustaw w config/config.yaml: mode: "embedded"

# Uruchomienie
make run

# Lub:
python -m orchestrator.main
```

---

## 🎤 Przykładowe komendy głosowe

### Docker Control
```
"Zrestartuj backend"
"Pokaż logi frontend 20 linii"
"Status kontenerów"
"Zatrzymaj bazę danych"
```

### Vision / Kamery
```
"Co widzisz?"
"Ile osób jest w kadrze?"
"Gdzie jest laptop?"
"Dodaj kamerę rtsp://192.168.1.100:554/stream"
"Lista kamer"
```

### IoT / Smart Home
```
"Jaka jest temperatura w salonie?"
"Włącz światło w kuchni"
"Ustaw termostat na 22 stopnie"
```

---

## 📊 Porównanie trybów

| Cecha | Docker (Enterprise) | Embedded (Jetson) |
|-------|---------------------|-------------------|
| **Deployment** | Docker Compose | Native Python |
| **Hardware** | Server/PC | Jetson/Edge |
| **RAM** | 8GB+ | 8GB |
| **GPU** | Optional | Required (CUDA) |
| **Cameras** | Multi (USB/RTSP/HTTP) | Multi + CSI |
| **Docker control** | ✅ | ❌ |
| **MQTT** | ✅ | Optional |
| **TensorRT** | Optional | ✅ Recommended |
| **Latency** | ~1s | ~700ms |
| **Offline** | Partial | ✅ Full |

---

## 🔧 Stack technologiczny

| Komponent | Technologia | Opis |
|-----------|-------------|------|
| **STT** | Faster-Whisper | Speech-to-Text, modele small/medium |
| **TTS** | Piper | Text-to-Speech, polski głos |
| **LLM** | Ollama + Phi-3 | Lokalny LLM 3.8B parametrów |
| **Vision** | YOLOv8 + TensorRT | Detekcja obiektów real-time |
| **Cameras** | OpenCV + GStreamer | Multi-source video capture |
| **MQTT** | Eclipse Mosquitto | IoT communication |
| **Docker** | Docker SDK | Container management |

---

## 🧪 Testy

```bash
# Uruchomienie wszystkich testów
make test

# Lub bezpośrednio:
pytest tests/ -v

# Z coverage
make test-cov
```

---

## 📄 Licencja

MIT License - Softreck / prototypowanie.pl

Używaj dowolnie w projektach komercyjnych i niekomercyjnych.

---

## 🤝 Wsparcie

- **GitHub Issues:** Zgłaszanie błędów i propozycji
- **Dokumentacja:** `/docs` w każdym projekcie
- **Kontakt:** hello@softreck.com
