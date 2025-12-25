"""
Tests for Speech-to-Text module
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestSpeechToText:
    """Tests for STT module."""
    
    @pytest.fixture
    def stt_config(self):
        return {
            "model": "tiny",  # Faster for tests
            "language": "pl",
            "beam_size": 1,
            "compute_type": "float32",
            "device": "cpu"
        }
    
    @pytest.fixture
    def audio_config(self):
        return {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "vad": {
                "enabled": False
            },
            "input_device": "auto"
        }
    
    def test_config_parsing(self, stt_config, audio_config):
        """Test configuration parsing."""
        from src.audio.stt import SpeechToText
        
        stt = SpeechToText(stt_config, audio_config)
        
        assert stt.model_name == "tiny"
        assert stt.language == "pl"
        assert stt.sample_rate == 16000
    
    @pytest.mark.asyncio
    async def test_initialization(self, stt_config, audio_config):
        """Test STT initialization."""
        from src.audio.stt import SpeechToText
        
        stt = SpeechToText(stt_config, audio_config)
        
        # Mock the model loading
        with patch('src.audio.stt.WhisperModel') as mock_whisper:
            mock_whisper.return_value = MagicMock()
            await stt.initialize()
            
            mock_whisper.assert_called_once_with(
                "tiny",
                device="cpu",
                compute_type="float32"
            )
    
    def test_vad_detection(self, stt_config, audio_config):
        """Test VAD configuration."""
        audio_config["vad"]["enabled"] = True
        audio_config["vad"]["mode"] = 3
        
        from src.audio.stt import SpeechToText
        
        stt = SpeechToText(stt_config, audio_config)
        
        assert stt.vad_enabled == True
        assert stt.vad_mode == 3


class TestAudioProcessing:
    """Tests for audio processing utilities."""
    
    def test_audio_normalization(self):
        """Test int16 to float32 conversion."""
        # Simulate int16 audio
        audio_int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        
        # Convert to float32
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        assert audio_float.dtype == np.float32
        assert -1.0 <= audio_float.min() <= 1.0
        assert -1.0 <= audio_float.max() <= 1.0
    
    def test_chunk_concatenation(self):
        """Test audio chunk concatenation."""
        chunks = [
            np.array([1, 2, 3], dtype=np.int16),
            np.array([4, 5, 6], dtype=np.int16),
            np.array([7, 8, 9], dtype=np.int16),
        ]
        
        combined = np.concatenate(chunks)
        
        assert len(combined) == 9
        assert combined[0] == 1
        assert combined[-1] == 9
