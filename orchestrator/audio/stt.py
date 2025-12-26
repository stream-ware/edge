"""
Speech-to-Text module for Orchestrator

Oparty na Faster-Whisper z VAD.
Kompatybilny z wersją Jetson.
Używa settings do autodetekcji GPU/CPU.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional
import queue
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..settings import settings

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None


class SpeechToText:
    """Streaming STT z VAD i autodetekcją GPU/CPU."""
    
    def __init__(self, stt_config: dict = None, audio_config: dict = None, on_speech_start=None):
        self.logger = logging.getLogger("stt")
        
        # Use settings as defaults, allow config overrides
        stt_config = stt_config or {}
        audio_config = audio_config or {}
        
        # Audio config (from settings or config)
        self.sample_rate = audio_config.get("sample_rate", settings.AUDIO_SAMPLE_RATE)
        self.channels = audio_config.get("channels", settings.AUDIO_CHANNELS)
        self.input_device = self._normalize_input_device(
            audio_config.get("input_device", settings.AUDIO_INPUT_DEVICE)
        )
        
        # VAD config
        vad_config = audio_config.get("vad", {})
        self.vad_enabled = vad_config.get("enabled", settings.AUDIO_VAD_ENABLED)
        self.vad_mode = vad_config.get("mode", settings.AUDIO_VAD_MODE)
        self.silence_duration = vad_config.get("silence_duration", settings.AUDIO_VAD_SILENCE_DURATION)
        self.max_buffer_seconds = vad_config.get("max_buffer_seconds", settings.AUDIO_VAD_MAX_BUFFER_SECONDS)
        
        # STT config with auto-detection
        self.model_name = stt_config.get("model", settings.AUDIO_STT_MODEL)
        self.language = stt_config.get("language", settings.AUDIO_STT_LANGUAGE)

        self.beam_size = int(stt_config.get("beam_size", 1))
        
        # Device auto-detection: "auto" -> check GPU availability
        cfg_device = stt_config.get("device", settings.AUDIO_STT_DEVICE)
        cfg_compute = stt_config.get("compute_type", settings.AUDIO_STT_COMPUTE_TYPE)
        
        if cfg_device == "auto":
            self.device = settings.get_effective_device()
            self.compute_type = settings.get_effective_compute_type()
            self.logger.info(f"Auto-detected device: {self.device}, compute_type: {self.compute_type}")
        else:
            self.device = cfg_device
            self.compute_type = cfg_compute if cfg_compute != "auto" else ("float16" if cfg_device == "cuda" else "float32")
        
        # Components
        self.model: Optional[WhisperModel] = None
        self.vad = None
        self._input_stream = None
        
        # Buffers
        self.speech_buffer: list = []
        self._audio_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._running = False

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        
        # State
        self.is_speaking = False
        self.silence_frames = 0
        self._on_speech_start = on_speech_start

        self.whisper_vad_filter = bool(
            stt_config.get("whisper_vad_filter", not self.vad_enabled)
        )

    def _normalize_input_device(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip()
            if not v or v.lower() == "auto":
                return None
            try:
                return int(v)
            except ValueError:
                return v
        return value
    
    async def initialize(self):
        """Inicjalizacja modelu."""
        if WhisperModel is None:
            self.logger.warning("faster-whisper not installed, STT disabled")
            return

        if sd is None:
            self.logger.warning("sounddevice not installed, audio streaming disabled")

        # Final safety check for CUDA - verify cuDNN before attempting
        effective_device = self.device
        effective_compute_type = self.compute_type

        if effective_device == "cuda" and not settings.is_cudnn_available():
            self.logger.warning("cuDNN not available, falling back to CPU")
            effective_device = "cpu"
            effective_compute_type = "float32"
        
        self.logger.info(f"Loading Whisper model: {self.model_name} (device={effective_device}, compute={effective_compute_type})")
        
        try:
            self.model = WhisperModel(
                self.model_name,
                device=effective_device,
                compute_type=effective_compute_type
            )
        except Exception as e:
            self.logger.warning(f"Model init failed ({effective_device}), falling back to CPU: {e}")
            self.model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="float32"
            )
        
        if self.vad_enabled and webrtcvad:
            self.vad = webrtcvad.Vad(self.vad_mode)
        
        self.logger.info("✅ STT initialized")
    
    async def cleanup(self):
        """Zwolnienie zasobów."""
        self._running = False
        if self._input_stream:
            try:
                self._input_stream.stop()
            except Exception:
                pass
            try:
                self._input_stream.close()
            except Exception:
                pass

        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=False)
    
    async def stream(self) -> AsyncIterator[str]:
        """Generator transkrypcji."""
        if not self.model or not sd:
            self.logger.warning("STT not available")
            return
        
        self._running = True
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                self.logger.warning(f"Audio status: {status}")
            try:
                self._audio_queue.put_nowait(indata.copy())
            except queue.Full:
                try:
                    self._audio_queue.get_nowait()
                except Exception:
                    return
                try:
                    self._audio_queue.put_nowait(indata.copy())
                except Exception:
                    return
        
        try:
            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=frame_size,
                device=self.input_device,
                callback=audio_callback
            )
            self._input_stream.start()
        except Exception as e:
            self.logger.error(f"Cannot open audio stream: {e}")
            return
        
        frames_per_second = self.sample_rate / frame_size
        silence_threshold = int(self.silence_duration * frames_per_second)
        max_buffer_frames = int(max(self.max_buffer_seconds, 0.0) * frames_per_second)
        
        while self._running:
            try:
                audio_chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk[:, 0]
            
            # VAD
            is_speech = True
            if self.vad:
                try:
                    is_speech = self.vad.is_speech(audio_chunk.tobytes(), self.sample_rate)
                except:
                    pass
            
            if is_speech:
                if not self.is_speaking and self._on_speech_start:
                    try:
                        result = self._on_speech_start()
                        if asyncio.iscoroutine(result):
                            asyncio.create_task(result)
                    except Exception:
                        pass
                self.silence_frames = 0
                self.speech_buffer.append(audio_chunk)
                self.is_speaking = True

                if max_buffer_frames > 0 and len(self.speech_buffer) >= max_buffer_frames:
                    transcript = await self._transcribe()
                    if transcript:
                        yield transcript
                    self.speech_buffer.clear()
            else:
                if self.is_speaking:
                    self.silence_frames += 1
                    self.speech_buffer.append(audio_chunk)
                    
                    if self.silence_frames >= silence_threshold:
                        transcript = await self._transcribe()
                        if transcript:
                            yield transcript
                        
                        self.is_speaking = False
                        self.silence_frames = 0
                        self.speech_buffer.clear()
                else:
                    self.speech_buffer.clear()
    
    async def _transcribe(self) -> Optional[str]:
        """Transkrypcja bufora."""
        if not self.speech_buffer:
            return None
        
        audio = np.concatenate(self.speech_buffer)
        audio_float = audio.astype(np.float32) / 32768.0
        
        if len(audio_float) < self.sample_rate * 0.3:
            return None
        
        loop = asyncio.get_event_loop()
        
        try:
            segments, _ = await loop.run_in_executor(
                self._executor,
                lambda: self.model.transcribe(
                    audio_float,
                    language=self.language if self.language != "auto" else None,
                    beam_size=self.beam_size,
                    vad_filter=self.whisper_vad_filter
                )
            )
            
            text = " ".join(s.text.strip() for s in segments).strip()
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None
