"""
Text-to-Speech module for Orchestrator

Oparty na Piper TTS z fallback do espeak.
"""

import asyncio
import logging
from typing import Optional
from pathlib import Path
import tempfile
import os
import threading

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except (ImportError, OSError):
    sd = None
    sf = None

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError:
    PiperVoice = None
    SynthesisConfig = None


class TextToSpeech:
    """Text-to-Speech z Piper."""
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger("tts")
        
        self.model_name = config.get("model", "pl_PL-gosia-medium")
        self.model_path = config.get("model_path", f"models/piper/{self.model_name}.onnx")
        self.config_path = config.get("config_path", f"models/piper/{self.model_name}.onnx.json")
        
        self.length_scale = config.get("length_scale", 1.0)
        self.sample_rate = config.get("sample_rate", 22050)
        
        self.voice: Optional[PiperVoice] = None
        self._use_cli = False
        self._is_speaking = False
        self._lock = asyncio.Lock()
        self._stop_requested = False
        self._playback_event = threading.Event()
        self._playback_lock = threading.Lock()
        self._current_proc = None
    
    async def initialize(self):
        """Inicjalizacja TTS."""
        if sd is None or sf is None:
            self.logger.warning("sounddevice/soundfile not installed, audio playback disabled")
        
        model_file = Path(self.model_path)
        
        if model_file.exists() and PiperVoice:
            try:
                self.voice = PiperVoice.load(
                    str(model_file),
                    str(Path(self.config_path))
                )
                self._use_cli = False
                self.logger.info("✅ Piper TTS loaded")
                return
            except Exception as e:
                self.logger.warning(f"Cannot load Piper model: {e}")
        
        self._use_cli = True
        self.logger.info("✅ TTS using CLI fallback")
    
    async def cleanup(self):
        """Zwolnienie zasobów."""
        self.voice = None

    async def stop(self):
        self._stop_requested = True

        with self._playback_lock:
            self._playback_event.set()

        proc = self._current_proc
        if proc is not None and getattr(proc, "returncode", None) is None:
            try:
                proc.terminate()
            except Exception:
                pass
    
    async def speak(self, text: str):
        """Synteza i odtworzenie tekstu."""
        if not text or not text.strip():
            return
        
        if sd is None:
            self.logger.warning("sounddevice not available")
            return
        
        async with self._lock:
            self._stop_requested = False
            with self._playback_lock:
                self._playback_event = threading.Event()
                playback_event = self._playback_event
            self._is_speaking = True
            
            try:
                self.logger.debug(f"TTS: {text[:50]}...")
                
                if not self._use_cli and self.voice:
                    await self._speak_native(text, playback_event)
                else:
                    await self._speak_cli(text, playback_event)
                    
            except Exception as e:
                self.logger.error(f"TTS error: {e}")
            finally:
                self._is_speaking = False
    
    async def _speak_native(self, text: str, playback_event: threading.Event):
        """Synteza przez Piper native."""
        loop = asyncio.get_event_loop()
        
        audio = await loop.run_in_executor(None, self._synthesize, text)

        if self._stop_requested or playback_event.is_set():
            return
        
        if audio is not None and len(audio) > 0:
            await loop.run_in_executor(
                None,
                lambda: self._play(audio, samplerate=self.sample_rate, playback_event=playback_event)
            )
    
    def _synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synteza tekstu."""
        try:
            audio_list = []
            
            # Configure synthesis
            syn_config = None
            if SynthesisConfig:
                syn_config = SynthesisConfig(length_scale=self.length_scale)
            
            # Synthesize chunks
            for chunk in self.voice.synthesize(text, syn_config=syn_config):
                audio_list.append(chunk.audio_int16_array)
            
            return np.concatenate(audio_list) if audio_list else None
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            return None
    
    def _play(self, audio: np.ndarray, samplerate: Optional[int] = None, playback_event: Optional[threading.Event] = None):
        """Odtwarzanie audio."""
        stream = None
        try:
            if sd is None:
                return

            stop_ev = playback_event
            if stop_ev is None:
                stop_ev = threading.Event()

            sr = int(samplerate or self.sample_rate)

            if audio is None or len(audio) == 0:
                return

            if audio.dtype.kind in {"f"}:
                audio_float = audio.astype(np.float32)
            else:
                audio_float = audio.astype(np.float32) / 32768.0

            if audio_float.ndim == 1:
                audio_float = audio_float.reshape(-1, 1)

            stream = sd.OutputStream(
                samplerate=sr,
                channels=int(audio_float.shape[1]),
                dtype="float32",
                blocksize=0,
            )
            stream.start()

            idx = 0
            block = 2048
            while idx < len(audio_float) and not stop_ev.is_set():
                stream.write(audio_float[idx: idx + block])
                idx += block

        except Exception as e:
            self.logger.error(f"Playback error: {e}")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
    
    async def _speak_cli(self, text: str, playback_event: threading.Event):
        """Fallback do CLI."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Próbuj Piper CLI
            cmd = f'echo "{text}" | piper --model {self.model_name} --output_file {temp_path} 2>/dev/null'
            proc = await asyncio.create_subprocess_shell(cmd)
            self._current_proc = proc
            await proc.communicate()
            self._current_proc = None
            
            if proc.returncode != 0 or not os.path.exists(temp_path):
                # Fallback do espeak
                cmd = f'espeak-ng -v pl "{text}" --stdout > {temp_path} 2>/dev/null'
                proc = await asyncio.create_subprocess_shell(cmd)
                self._current_proc = proc
                await proc.communicate()
                self._current_proc = None

            if self._stop_requested or playback_event.is_set():
                return
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                data, rate = sf.read(temp_path)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._play(data, samplerate=rate, playback_event=playback_event)
                )
                
        except Exception as e:
            self.logger.error(f"CLI TTS error: {e}")
        finally:
            self._current_proc = None
            try:
                os.unlink(temp_path)
            except:
                pass
    
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
