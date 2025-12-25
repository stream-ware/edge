"""
Tests for Text-to-Speech module
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock


class TestTextToSpeech:
    """Tests for TTS module."""
    
    @pytest.fixture
    def tts_config(self):
        return {
            "model": "pl_PL-gosia-medium",
            "model_path": "models/piper/pl_PL-gosia-medium.onnx",
            "config_path": "models/piper/pl_PL-gosia-medium.onnx.json",
            "speaker_id": 0,
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_w": 0.8,
            "sample_rate": 22050
        }
    
    def test_config_parsing(self, tts_config):
        """Test configuration parsing."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts = TextToSpeech(tts_config)
        
        assert tts.model_name == "pl_PL-gosia-medium"
        assert tts.length_scale == 1.0
        assert tts.sample_rate == 22050
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self, tts_config):
        """Test handling of empty text."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts = TextToSpeech(tts_config)
        tts._use_cli = True  # Skip model loading
        
        # Should not raise and return quickly
        await tts.speak("")
        await tts.speak("   ")
        await tts.speak(None)  # type: ignore
    
    @pytest.mark.asyncio
    async def test_speaking_state(self, tts_config):
        """Test is_speaking property."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts = TextToSpeech(tts_config)
        
        # Initially not speaking
        assert tts.is_speaking == False


class TestTTSFallback:
    """Tests for TTS fallback mechanisms."""
    
    @pytest.fixture
    def tts_config(self):
        return {
            "model": "pl_PL-gosia-medium",
            "model_path": "nonexistent.onnx",
        }
    
    @pytest.mark.asyncio
    async def test_cli_fallback(self, tts_config):
        """Test fallback to CLI mode when model not found."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts = TextToSpeech(tts_config)
        await tts.initialize()
        
        # Should use CLI mode
        assert tts._use_cli == True
