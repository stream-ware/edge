# 📦 Instalacja Streamware

## Wymagania systemowe

### Minimalne

| Komponent | Wymaganie |
|-----------|-----------|
| OS | Ubuntu 22.04+ / Debian 12+ |
| RAM | 8GB |
| Storage | 20GB |
| Python | 3.10+ |

### Zalecane (dla Vision)

| Komponent | Wymaganie |
|-----------|-----------|
| GPU | NVIDIA z 6GB+ VRAM |
| CUDA | 11.8+ |
| Jetson | JetPack 6.0+ |

---

## 1. Streamware Orchestrator (Docker)

### 1.1 Szybka instalacja

```bash
# Klonuj repo
git clone https://github.com/softreck/streamware.git
cd streamware/streamware-orchestrator

# Uruchom (development)
docker-compose -f docker-compose-single.yml up -d

# Lub production
docker-compose -f docker-compose-full.yml up -d
```

### 1.2 Instalacja manualna

```bash
# System dependencies
sudo apt update
sudo apt install -y \
    python3-pip python3-venv \
    portaudio19-dev libsndfile1-dev \
    ffmpeg espeak-ng \
    docker.io docker-compose

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini

# Python environment
cd streamware-orchestrator/orchestrator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Uruchom
python main.py
```

### 1.3 Docker Compose Profiles

```bash
# Development (MQTT + Orchestrator)
docker-compose -f docker-compose-single.yml up

# Staging (+ Database + Backend)
docker-compose -f docker-compose-multi.yml up

# Production (+ IoT simulator + Monitoring)
docker-compose -f docker-compose-full.yml up

# Z monitoringiem (Prometheus + Grafana)
docker-compose -f docker-compose-full.yml --profile monitoring up
```

---

## 2. Streamware Jetson (Native)

### 2.1 Automatyczna instalacja

```bash
cd streamware/streamware-jetson
chmod +x scripts/install.sh
./scripts/install.sh
```

### 2.2 Instalacja krok po kroku

#### System dependencies

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
    python3-pip python3-venv \
    portaudio19-dev libsndfile1 \
    libopencv-dev ffmpeg \
    espeak-ng libespeak-ng-dev
```

#### Python environment

```bash
cd streamware-jetson
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama
ollama pull phi3:mini
```

#### Piper TTS (polski głos)

```bash
./scripts/download_piper_pl.sh medium gosia
```

#### YOLOv8 TensorRT (opcjonalne, zalecane)

```bash
python scripts/export_yolo_tensorrt.py --model yolov8n
```

### 2.3 Uruchomienie

```bash
source venv/bin/activate
python main.py
```

---

## 3. Modele AI

### 3.1 LLM (Ollama)

```bash
# Phi-3 Mini (3.8B) - zalecany
ollama pull phi3:mini

# Alternatywy
ollama pull llama3.2:1b    # Mniejszy, szybszy
ollama pull gemma2:2b      # Dobra jakość
ollama pull mistral:7b     # Większy, lepszy
```

### 3.2 Whisper (STT)

Modele pobierają się automatycznie przy pierwszym uruchomieniu.

| Model | Rozmiar | VRAM | Jakość |
|-------|---------|------|--------|
| tiny | 75MB | ~1GB | Niska |
| base | 150MB | ~1GB | Średnia |
| small | 500MB | ~2GB | **Zalecana** |
| medium | 1.5GB | ~4GB | Wysoka |

### 3.3 Piper TTS

```bash
# Polski głos (gosia)
./scripts/download_piper_pl.sh medium gosia

# Lub ręcznie
cd models/piper
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json
```

### 3.4 YOLOv8

```bash
# PyTorch (automatyczne pobieranie)
# Po prostu uruchom - model pobierze się sam

# TensorRT (Jetson - zalecane)
python scripts/export_yolo_tensorrt.py --model yolov8n
```

---

## 4. Hardware Setup

### 4.1 Mikrofon USB

```bash
# Sprawdź urządzenia
arecord -l

# Test
arecord -d 5 -f cd test.wav
aplay test.wav

# Jeśli nie działa
sudo usermod -aG audio $USER
# Wyloguj i zaloguj ponownie
```

### 4.2 Kamera USB

```bash
# Sprawdź urządzenia
ls -la /dev/video*
v4l2-ctl --list-devices

# Test
ffplay /dev/video0

# Uprawnienia
sudo usermod -aG video $USER
```

### 4.3 Kamera RTSP

```bash
# Test streamu
ffplay "rtsp://192.168.1.100:554/stream"

# Z hasłem
ffplay "rtsp://admin:password@192.168.1.100:554/stream"
```

### 4.4 Jetson Power Mode

```bash
# Maksymalna wydajność
sudo nvpmodel -m 0
sudo jetson_clocks

# Sprawdź status
sudo nvpmodel -q
```

---

## 5. Weryfikacja instalacji

### 5.1 Test komponentów

```bash
# STT
python -c "from faster_whisper import WhisperModel; print('STT OK')"

# TTS
python -c "import piper; print('TTS OK')"

# Vision
python -c "from ultralytics import YOLO; print('Vision OK')"

# LLM
curl http://localhost:11434/api/tags
```

### 5.2 Benchmark

```bash
# Orchestrator
cd streamware-orchestrator/orchestrator
python -m pytest tests/

# Jetson
cd streamware-jetson
python scripts/benchmark.py
```

### 5.3 Test end-to-end

```bash
# Uruchom system
python main.py

# Powiedz: "Pomoc"
# System powinien odpowiedzieć listą dostępnych komend
```

---

## 6. Rozwiązywanie problemów

### CUDA not found

```bash
# Sprawdź CUDA
nvidia-smi
nvcc --version

# Jeśli brak - zainstaluj
# Ubuntu: https://developer.nvidia.com/cuda-downloads
```

### Audio permission denied

```bash
sudo usermod -aG audio $USER
# Wyloguj i zaloguj
```

### Docker permission denied

```bash
sudo usermod -aG docker $USER
# Wyloguj i zaloguj
```

### Ollama not running

```bash
sudo systemctl start ollama
sudo systemctl enable ollama

# Sprawdź status
sudo systemctl status ollama
```

### RTSP connection timeout

```bash
# Sprawdź sieć
ping 192.168.1.100

# Sprawdź port
nc -zv 192.168.1.100 554

# Próbuj różne transporty w config:
rtsp_transport: "udp"  # zamiast "tcp"
```
