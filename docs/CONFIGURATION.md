# ⚙️ Konfiguracja Streamware

## Plik konfiguracyjny

Główny plik konfiguracyjny: `config/config.yaml`

---

## 1. Audio

```yaml
audio:
  # Włącz/wyłącz audio
  enabled: true
  
  # Parametry nagrywania
  sample_rate: 16000      # Hz (16000 dla Whisper)
  channels: 1             # Mono
  chunk_size: 1024        # Samples per chunk
  
  # Urządzenie (auto lub numer)
  input_device: "auto"    # lub 0, 1, "hw:1,0"
  output_device: "auto"
  
  # Voice Activity Detection
  vad:
    enabled: true
    mode: 3               # 0-3 (wyższy = bardziej agresywny)
    threshold: 0.6        # Próg detekcji
    min_speech_duration: 0.5  # Min. czas mowy (s)
    silence_duration: 0.8     # Cisza = koniec (s)
```

### STT (Speech-to-Text)

```yaml
  stt:
    model: "small"        # tiny, base, small, medium, large
    language: "pl"        # pl, en, auto
    beam_size: 5          # Beam search width
    compute_type: "float16"  # float16, float32, int8
    device: "cuda"        # cuda, cpu
    
    # Opcje zaawansowane
    vad_filter: true
    word_timestamps: false
    condition_on_previous_text: true
```

### TTS (Text-to-Speech)

```yaml
  tts:
    model: "pl_PL-gosia-medium"
    model_path: "models/piper/pl_PL-gosia-medium.onnx"
    config_path: "models/piper/pl_PL-gosia-medium.onnx.json"
    
    # Parametry głosu
    speaker_id: 0
    length_scale: 1.0     # Prędkość (0.5=szybciej, 2.0=wolniej)
    noise_scale: 0.667
    noise_w: 0.8
    sample_rate: 22050
```

---

## 2. LLM

```yaml
llm:
  provider: "ollama"
  model: "phi3:mini"      # phi3:mini, llama3.2:1b, gemma2:2b
  base_url: "http://localhost:11434"
  
  # Parametry generowania
  temperature: 0.7        # Kreatywność (0.0-2.0)
  max_tokens: 256         # Max długość odpowiedzi
  top_p: 0.9              # Nucleus sampling
  
  # Timeout
  timeout: 30.0           # Sekundy
  
  # System prompt
  system_prompt: |
    Jesteś pomocnym asystentem.
    Odpowiadaj krótko po polsku.
```

---

## 3. Vision

```yaml
vision:
  # Model detekcji
  model: "yolov8n"        # yolov8n, yolov8s, yolov8m
  model_path: null        # Ścieżka do TensorRT engine (opcjonalne)
  
  # Parametry detekcji
  confidence: 0.5         # Min. pewność (0.0-1.0)
  iou_threshold: 0.45     # IoU dla NMS
  max_detections: 20      # Max obiektów na klatkę
  
  # Przetwarzanie
  process_every_n_frames: 5  # Co N-ta klatka
  translate_labels: true     # Tłumacz na polski
  
  # Klasy do wykrywania (null = wszystkie)
  classes: null           # lub [0, 1, 2] dla person, bicycle, car
```

### Kamery

```yaml
  cameras:
    # USB camera
    - source: 0
      name: "usb_cam"
      width: 640
      height: 480
      fps: 30
      use_gstreamer: false
    
    # RTSP camera
    - source: "rtsp://192.168.1.100:554/stream"
      name: "ip_camera"
      width: 1280
      height: 720
      fps: 30
      rtsp_transport: "tcp"   # tcp, udp, http
      use_gstreamer: true
      use_hw_decode: true     # Hardware decode (Jetson)
      
      # Reconnect
      reconnect: true
      reconnect_delay: 5.0
      max_reconnect_attempts: 10
    
    # HTTP MJPEG
    - source: "http://192.168.1.101/video.mjpg"
      name: "webcam"
    
    # CSI camera (Jetson)
    - source: "csi://0"
      name: "csi_cam"
      use_gstreamer: true
```

---

## 4. MQTT

```yaml
mqtt:
  enabled: true
  broker: "localhost"     # lub "mqtt" w Docker
  port: 1883
  
  # Autentykacja (opcjonalne)
  username: null
  password: null
  
  client_id: "streamware-orchestrator"
  
  # Topics
  topics:
    commands: "commands/#"
    events: "events/#"
    sensors: "edge/sensors"
    tts: "audio/tts"
    stt: "audio/stt"
```

---

## 5. Docker

```yaml
docker:
  socket: "unix:///var/run/docker.sock"
  timeout: 30
  
  # Dla remote Docker
  # host: "tcp://192.168.1.100:2375"
```

---

## 6. Adaptery

```yaml
adapters:
  enabled:
    - docker
    - mqtt
    - firmware
    - vision
  
  # Firmware/IoT
  firmware:
    simulation: true      # Symulowane sensory
```

---

## 7. Orchestrator

```yaml
orchestrator:
  # Tryb działania
  mode: "continuous"      # continuous, push_to_talk, wake_word
  
  # Wake word (dla mode=wake_word)
  wake_word: "hej asystent"
  
  # Timeout bezczynności
  idle_timeout: 300       # Sekundy
  
  # Bufor (dla trybu z nagrywaniem)
  buffer:
    enabled: false
    audio_seconds: 30
    video_frames: 150
```

---

## 8. Logging

```yaml
logging:
  level: "INFO"           # DEBUG, INFO, WARNING, ERROR
  file: "logs/streamware.log"
  max_size: "10MB"
  backup_count: 3
  
  # Per-component levels
  components:
    stt: "INFO"
    vision: "INFO"
    llm: "INFO"
    tts: "INFO"
    orchestrator: "DEBUG"
```

---

## 9. Performance (Jetson)

```yaml
performance:
  # GPU memory
  gpu_memory_fraction: 0.8
  
  # CPU priority
  use_realtime_priority: false
  
  # Jetson power mode (0=MAX, 1=15W, 2=7W)
  power_mode: 0
```

---

## Zmienne środowiskowe

Możesz nadpisać wartości przez env vars:

```bash
# MQTT
export MQTT_BROKER=mqtt.example.com
export MQTT_PORT=1883

# Ollama
export OLLAMA_HOST=http://localhost:11434

# Database
export DB_HOST=postgres.example.com
export DB_PASSWORD=secret

# Secrets (nie wpisuj do config.yaml!)
export HA_TOKEN=your_home_assistant_token
```

W `config.yaml` użyj `${VAR_NAME}`:

```yaml
homeassistant:
  token: "${HA_TOKEN}"
```

---

## Profile konfiguracji

### Development

```yaml
audio:
  stt:
    model: "tiny"         # Szybszy
logging:
  level: "DEBUG"
```

### Production

```yaml
audio:
  stt:
    model: "small"
logging:
  level: "WARNING"
performance:
  power_mode: 0           # MAX performance
```

### Low-power (Edge)

```yaml
audio:
  stt:
    model: "tiny"
    device: "cpu"
vision:
  process_every_n_frames: 10
performance:
  power_mode: 2           # 7W mode
```
