"""
Speech-to-Text module for Orchestrator

Oparty na Faster-Whisper z VAD.
Kompatybilny z wersją Jetson.
"""

import asyncio
import logging
from typing import AsyncIterator, Optional
import queue
import numpy as np

try:
    import sounddevice as sd
except ImportError:
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
    """Streaming STT z VAD."""
    
    def __init__(self, stt_config: dict, audio_config: dict):
        self.logger = logging.getLogger("stt")
        
        # Audio config
        self.sample_rate = audio_config.get("sample_rate", 16000)
        self.channels = audio_config.get("channels", 1)
        
        # VAD config
        vad_config = audio_config.get("vad", {})
        self.vad_enabled = vad_config.get("enabled", True)
        self.vad_mode = vad_config.get("mode", 3)
        self.silence_duration = vad_config.get("silence_duration", 0.8)
        
        # STT config
        self.model_name = stt_config.get("model", "small")
        self.language = stt_config.get("language", "pl")
        self.compute_type = stt_config.get("compute_type", "float16")
        self.device = stt_config.get("device", "cuda")
        
        # Components
        self.model: Optional[WhisperModel] = None
        self.vad = None
        self.stream = None
        
        # Buffers
        self.speech_buffer: list = []
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        
        # State
        self.is_speaking = False
        self.silence_frames = 0
    
    async def initialize(self):
        """Inicjalizacja modelu."""
        if WhisperModel is None:
            self.logger.warning("faster-whisper not installed, STT disabled")
            return

        if sd is None:
            self.logger.warning("sounddevice not installed, audio streaming disabled")
        
        self.logger.info(f"Loading Whisper model: {self.model_name}")
        
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            self.logger.warning(f"GPU init failed, falling back to CPU: {e}")
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
        if self.stream:
            self.stream.stop()
            self.stream.close()
    
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
            self._audio_queue.put(indata.copy())
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
                blocksize=frame_size,
                callback=audio_callback
            )
            self.stream.start()
        except Exception as e:
            self.logger.error(f"Cannot open audio stream: {e}")
            return
        
        frames_per_second = self.sample_rate / frame_size
        silence_threshold = int(self.silence_duration * frames_per_second)
        
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
                self.silence_frames = 0
                self.speech_buffer.append(audio_chunk)
                self.is_speaking = True
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
                None,
                lambda: self.model.transcribe(
                    audio_float,
                    language=self.language if self.language != "auto" else None,
                    beam_size=5,
                    vad_filter=True
                )
            )
            
            text = " ".join(s.text.strip() for s in segments).strip()
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None
