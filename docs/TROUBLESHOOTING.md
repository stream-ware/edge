# 🔧 Troubleshooting Guide

## 1. Audio Problems

### 1.1 Mikrofon nie działa

**Symptom:** Brak rozpoznawania mowy, cisza w logach STT

**Rozwiązania:**

```bash
# Sprawdź urządzenia
arecord -l

# Test nagrywania
arecord -d 5 -f cd test.wav
aplay test.wav

# Uprawnienia
sudo usermod -aG audio $USER
# Wyloguj i zaloguj ponownie

# Sprawdź domyślne urządzenie
pactl info | grep "Default Source"

# Ustaw w config.yaml
audio:
  input_device: 1  # numer z arecord -l
```

### 1.2 TTS nie mówi

**Symptom:** Brak dźwięku po odpowiedzi

**Rozwiązania:**

```bash
# Sprawdź głośniki
speaker-test -t wav

# Test Piper
echo "Test" | piper --model pl_PL-gosia-medium --output_file test.wav
aplay test.wav

# Sprawdź czy model istnieje
ls -la models/piper/

# Pobierz model jeśli brak
./scripts/download_piper_pl.sh

# Fallback do espeak
sudo apt install espeak-ng
espeak-ng -v pl "Test"
```

### 1.3 VAD zbyt czuły/nieczuły

**Symptom:** Przerywa w środku zdania / nie wykrywa początku mowy

```yaml
# config.yaml - dostosuj
audio:
  vad:
    mode: 2              # 0=łagodny, 3=agresywny
    threshold: 0.5       # Niższy = czulszy
    silence_duration: 1.0  # Dłuższy = więcej ciszy przed końcem
    min_speech_duration: 0.3  # Krótszy = szybsza reakcja
```

---

## 2. Vision Problems

### 2.1 Kamera USB nie działa

**Symptom:** "Cannot open camera" w logach

```bash
# Sprawdź urządzenia
ls -la /dev/video*
v4l2-ctl --list-devices

# Test z ffmpeg
ffplay /dev/video0

# Uprawnienia
sudo usermod -aG video $USER

# Sprawdź czy w użyciu przez inny proces
fuser /dev/video0

# W config.yaml spróbuj inny numer
vision:
  cameras:
    - source: 1  # zamiast 0
```

### 2.2 RTSP nie łączy się

**Symptom:** Timeout, reconnect loop

```bash
# Test streamu
ffplay "rtsp://192.168.1.100:554/stream"

# Z hasłem
ffplay "rtsp://admin:password@192.168.1.100:554/stream"

# Sprawdź sieć
ping 192.168.1.100
nc -zv 192.168.1.100 554

# W config.yaml spróbuj inny transport
cameras:
  - source: "rtsp://..."
    rtsp_transport: "udp"  # zamiast "tcp"
```

### 2.3 Wysokie opóźnienie RTSP

```yaml
# Zmniejsz bufor
cameras:
  - source: "rtsp://..."
    rtsp_buffer_size: 512000  # Mniejszy bufor
    
# Użyj GStreamer z niższym latency
    use_gstreamer: true
```

### 2.4 YOLO wolny / out of memory

```yaml
# Użyj mniejszego modelu
vision:
  model: "yolov8n"  # nano zamiast small/medium

# Zmniejsz częstotliwość
  process_every_n_frames: 10  # Co 10 klatek

# Zmniejsz rozdzielczość
  cameras:
    - source: 0
      width: 416
      height: 416

# Użyj TensorRT (Jetson)
python scripts/export_yolo_tensorrt.py
```

### 2.5 GStreamer nie działa

```bash
# Instalacja
sudo apt install \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad

# Test pipeline
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink

# Jeśli nie działa, wyłącz w config
use_gstreamer: false
```

---

## 3. LLM Problems

### 3.1 Ollama nie odpowiada

**Symptom:** Timeout w logach LLM

```bash
# Sprawdź czy działa
curl http://localhost:11434/api/tags

# Uruchom jeśli nie działa
sudo systemctl start ollama

# Sprawdź logi
journalctl -u ollama -f

# Restart
sudo systemctl restart ollama
```

### 3.2 Model nie załadowany

```bash
# Lista modeli
ollama list

# Pobierz model
ollama pull phi3:mini

# Test
ollama run phi3:mini "Cześć"
```

### 3.3 Wolna odpowiedź LLM

```yaml
# Użyj mniejszego modelu
llm:
  model: "llama3.2:1b"  # zamiast phi3:mini

# Zmniejsz max_tokens
  max_tokens: 128

# Zmniejsz temperature (szybsze sampling)
  temperature: 0.3
```

### 3.4 GPU out of memory (LLM)

```bash
# Sprawdź zużycie
nvidia-smi

# Użyj mniejszego modelu
ollama pull llama3.2:1b

# Lub quantized
ollama pull phi3:mini-q4
```

---

## 4. Docker Problems

### 4.1 Permission denied

```bash
# Dodaj do grupy docker
sudo usermod -aG docker $USER
# Wyloguj i zaloguj

# Lub uruchom z sudo
sudo python main.py
```

### 4.2 Socket not found

```yaml
# Sprawdź ścieżkę
docker:
  socket: "unix:///var/run/docker.sock"

# Dla remote Docker
  socket: "tcp://192.168.1.100:2375"
```

### 4.3 Container not found

**Symptom:** "Container 'xyz' not found"

```bash
# Lista kontenerów (wszystkich)
docker ps -a

# Sprawdź nazwę
docker inspect backend --format '{{.Name}}'
```

---

## 5. MQTT Problems

### 5.1 Connection refused

```bash
# Sprawdź czy broker działa
sudo systemctl status mosquitto

# Uruchom
sudo systemctl start mosquitto

# W Docker
docker-compose logs mqtt
```

### 5.2 Authentication failed

```yaml
# W config.yaml dodaj credentials
mqtt:
  username: "user"
  password: "pass"
```

### 5.3 Messages not received

```bash
# Test subskrypcji
mosquitto_sub -t "commands/#" -v

# W innym terminalu publikuj
mosquitto_pub -t "commands/test" -m "hello"
```

---

## 6. Performance Issues

### 6.1 Wysokie zużycie CPU

```yaml
# Zmniejsz częstotliwość vision
vision:
  process_every_n_frames: 15

# Użyj GPU dla STT
audio:
  stt:
    device: "cuda"
```

### 6.2 Wysokie zużycie RAM

```yaml
# Mniejsze modele
audio:
  stt:
    model: "tiny"

vision:
  model: "yolov8n"

# Mniej kamer
cameras:
  - source: 0  # tylko jedna
```

### 6.3 Jetson overheating

```bash
# Sprawdź temperaturę
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Niższy power mode
sudo nvpmodel -m 1  # 15W zamiast MAX

# Lub
sudo nvpmodel -m 2  # 7W
```

---

## 7. Common Errors

### 7.1 "ModuleNotFoundError"

```bash
# Aktywuj venv
source venv/bin/activate

# Zainstaluj brakujące
pip install <module>

# Lub wszystko
pip install -r requirements.txt
```

### 7.2 "CUDA out of memory"

```bash
# Sprawdź co używa GPU
nvidia-smi

# Zabij inne procesy
kill <pid>

# Użyj CPU
device: "cpu"
```

### 7.3 "Address already in use"

```bash
# Znajdź proces na porcie
lsof -i :1883
lsof -i :11434

# Zabij
kill <pid>
```

### 7.4 "Permission denied" (files)

```bash
# Napraw uprawnienia
chmod +x scripts/*.sh
chmod -R 755 models/
```

---

## 8. Debug Mode

### 8.1 Włącz verbose logging

```yaml
logging:
  level: "DEBUG"
  
  components:
    stt: "DEBUG"
    vision: "DEBUG"
    llm: "DEBUG"
```

### 8.2 Test pojedynczych komponentów

```bash
# STT
python -c "
from audio import SpeechToText
import asyncio
stt = SpeechToText({'model':'tiny'}, {'sample_rate':16000})
asyncio.run(stt.initialize())
print('STT OK')
"

# Vision
python -c "
from vision import ObjectDetector
import asyncio
det = ObjectDetector({'model':'yolov8n'})
asyncio.run(det.initialize())
print('Vision OK')
"

# LLM
curl -X POST http://localhost:11434/api/chat \
  -d '{"model":"phi3:mini","messages":[{"role":"user","content":"test"}]}'
```

---

## 9. Getting Help

### 9.1 Zbierz informacje diagnostyczne

```bash
# System
uname -a
cat /etc/os-release

# Python
python --version
pip list | grep -E "(faster-whisper|ultralytics|piper|ollama)"

# GPU
nvidia-smi

# Docker
docker --version
docker-compose --version

# Logi
tail -100 logs/streamware.log
```

### 9.2 Zgłoś issue

1. Opisz problem
2. Dołącz kroki do reprodukcji
3. Dołącz logi (DEBUG level)
4. Dołącz konfigurację (bez haseł!)
5. Dołącz informacje o systemie
