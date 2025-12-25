"""Audio module - STT and TTS"""
 
from typing import Any
 
__all__ = ["SpeechToText", "TextToSpeech"]
 
 
def __getattr__(name: str) -> Any:
    if name == "SpeechToText":
        from .stt import SpeechToText
 
        return SpeechToText
 
    if name == "TextToSpeech":
        from .tts import TextToSpeech
 
        return TextToSpeech
 
    raise AttributeError(name)
