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

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    sd = None
    sf = None

try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None


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
    
    async def initialize(self):
        """Inicjalizacja TTS."""
        if sd is None:
            self.logger.warning("sounddevice not installed, TTS disabled")
            return
        
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
    
    async def speak(self, text: str):
        """Synteza i odtworzenie tekstu."""
        if not text or not text.strip():
            return
        
        if sd is None:
            self.logger.warning("sounddevice not available")
            return
        
        async with self._lock:
            self._is_speaking = True
            
            try:
                self.logger.debug(f"TTS: {text[:50]}...")
                
                if not self._use_cli and self.voice:
                    await self._speak_native(text)
                else:
                    await self._speak_cli(text)
                    
            except Exception as e:
                self.logger.error(f"TTS error: {e}")
            finally:
                self._is_speaking = False
    
    async def _speak_native(self, text: str):
        """Synteza przez Piper native."""
        loop = asyncio.get_event_loop()
        
        audio = await loop.run_in_executor(None, self._synthesize, text)
        
        if audio is not None and len(audio) > 0:
            await loop.run_in_executor(None, self._play, audio)
    
    def _synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synteza tekstu."""
        try:
            audio_list = []
            for chunk in self.voice.synthesize_stream_raw(
                text,
                length_scale=self.length_scale
            ):
                audio_list.append(np.frombuffer(chunk, dtype=np.int16))
            
            return np.concatenate(audio_list) if audio_list else None
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            return None
    
    def _play(self, audio: np.ndarray):
        """Odtwarzanie audio."""
        try:
            audio_float = audio.astype(np.float32) / 32768.0
            sd.play(audio_float, samplerate=self.sample_rate)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
    
    async def _speak_cli(self, text: str):
        """Fallback do CLI."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Próbuj Piper CLI
            cmd = f'echo "{text}" | piper --model {self.model_name} --output_file {temp_path} 2>/dev/null'
            proc = await asyncio.create_subprocess_shell(cmd)
            await proc.communicate()
            
            if proc.returncode != 0 or not os.path.exists(temp_path):
                # Fallback do espeak
                cmd = f'espeak-ng -v pl "{text}" --stdout > {temp_path} 2>/dev/null'
                proc = await asyncio.create_subprocess_shell(cmd)
                await proc.communicate()
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                data, rate = sf.read(temp_path)
                sd.play(data, rate)
                sd.wait()
                
        except Exception as e:
            self.logger.error(f"CLI TTS error: {e}")
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
