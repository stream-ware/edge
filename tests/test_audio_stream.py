"""
Tests for Audio Streaming (STT/TTS integration)
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import queue


class TestSTTStream:
    """Tests for STT streaming functionality."""
    
    @pytest.fixture
    def stt_config(self):
        return {
            "model": "tiny",
            "language": "pl",
            "device": "cpu",
            "compute_type": "float32"
        }
    
    @pytest.fixture
    def audio_config(self):
        return {
            "sample_rate": 16000,
            "channels": 1,
            "vad": {
                "enabled": True,
                "mode": 3,
                "silence_duration": 0.5,
                "max_buffer_seconds": 5.0
            }
        }
    
    def test_stt_config_defaults(self, stt_config, audio_config):
        """Test STT uses settings defaults correctly."""
        from orchestrator.audio.stt import SpeechToText
        
        stt = SpeechToText(stt_config, audio_config)
        
        assert stt.sample_rate == 16000
        assert stt.channels == 1
        assert stt.vad_enabled == True
        assert stt.vad_mode == 3
        assert stt.silence_duration == 0.5
        assert stt.max_buffer_seconds == 5.0
    
    def test_stt_auto_device_detection(self):
        """Test auto device detection from settings."""
        from orchestrator.settings import settings
        
        # Should return 'cuda' or 'cpu' based on availability
        device = settings.get_effective_device()
        assert device in ('cuda', 'cpu')
        
        compute = settings.get_effective_compute_type()
        assert compute in ('float16', 'float32', 'int8')
    
    @pytest.mark.asyncio
    async def test_stt_stream_without_model(self, stt_config, audio_config):
        """Test stream gracefully handles missing model."""
        from orchestrator.audio.stt import SpeechToText
        
        stt = SpeechToText(stt_config, audio_config)
        # Don't initialize - model is None
        
        results = []
        async for transcript in stt.stream():
            results.append(transcript)
        
        # Should return immediately with warning
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_stt_cleanup(self, stt_config, audio_config):
        """Test STT cleanup releases resources."""
        from orchestrator.audio.stt import SpeechToText
        
        stt = SpeechToText(stt_config, audio_config)
        stt._running = True
        stt._input_stream = MagicMock()
        
        await stt.cleanup()
        
        assert stt._running == False
        stt._input_stream.stop.assert_called_once()
        stt._input_stream.close.assert_called_once()


class TestTTSStream:
    """Tests for TTS streaming functionality."""
    
    @pytest.fixture
    def tts_config(self):
        return {
            "model": "pl_PL-gosia-medium",
            "model_path": "models/piper/pl_PL-gosia-medium.onnx",
            "length_scale": 1.0,
            "sample_rate": 22050
        }
    
    def test_tts_config_parsing(self, tts_config):
        """Test TTS config parsing."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts = TextToSpeech(tts_config)
        
        assert tts.model_name == "pl_PL-gosia-medium"
        assert tts.length_scale == 1.0
        assert tts.sample_rate == 22050
    
    @pytest.mark.asyncio
    async def test_tts_speak_empty(self, tts_config):
        """Test TTS handles empty text."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts = TextToSpeech(tts_config)
        
        # Should not raise
        await tts.speak("")
        await tts.speak("   ")
    
    @pytest.mark.asyncio
    async def test_tts_cli_fallback(self, tts_config):
        """Test TTS falls back to CLI when model missing."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts_config["model_path"] = "/nonexistent/path.onnx"
        tts = TextToSpeech(tts_config)
        
        await tts.initialize()
        
        assert tts._use_cli == True
    
    @pytest.mark.asyncio
    async def test_tts_cleanup(self, tts_config):
        """Test TTS cleanup."""
        from orchestrator.audio.tts import TextToSpeech
        
        tts = TextToSpeech(tts_config)
        tts.voice = MagicMock()
        
        await tts.cleanup()
        
        assert tts.voice is None


class TestAudioBuffer:
    """Tests for audio buffer management."""
    
    def test_buffer_accumulation(self):
        """Test audio chunks accumulate correctly."""
        buffer = []
        
        for i in range(10):
            chunk = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
            buffer.append(chunk)
        
        combined = np.concatenate(buffer)
        assert len(combined) == 4800
    
    def test_buffer_max_size_flush(self):
        """Test buffer flushes at max size."""
        max_frames = 100
        buffer = []
        flushed = False
        
        for i in range(150):
            buffer.append(np.zeros(480, dtype=np.int16))
            
            if len(buffer) >= max_frames:
                # Simulate flush
                combined = np.concatenate(buffer)
                buffer.clear()
                flushed = True
                break
        
        assert flushed == True
        assert len(buffer) == 0
    
    def test_silence_detection_threshold(self):
        """Test silence frame counting."""
        silence_duration = 0.8
        frames_per_second = 16000 / 480  # ~33.3 fps
        silence_threshold = int(silence_duration * frames_per_second)
        
        assert silence_threshold == 26  # ~0.8s of silence


class TestVADIntegration:
    """Tests for VAD integration."""
    
    def test_vad_frame_size(self):
        """Test VAD requires specific frame sizes."""
        sample_rate = 16000
        
        # WebRTC VAD supports 10ms, 20ms, 30ms frames
        valid_frame_ms = [10, 20, 30]
        
        for ms in valid_frame_ms:
            frame_size = int(sample_rate * ms / 1000)
            assert frame_size in [160, 320, 480]
    
    def test_vad_mode_range(self):
        """Test VAD mode is in valid range."""
        valid_modes = [0, 1, 2, 3]
        
        for mode in valid_modes:
            assert 0 <= mode <= 3


class TestSettingsIntegration:
    """Tests for settings module integration."""
    
    def test_settings_singleton(self):
        """Test settings is singleton."""
        from orchestrator.settings import settings as s1
        from orchestrator.settings import settings as s2
        
        assert s1 is s2
    
    def test_settings_audio_defaults(self):
        """Test audio settings have defaults."""
        from orchestrator.settings import settings
        
        assert settings.AUDIO_SAMPLE_RATE == 16000
        assert settings.AUDIO_CHANNELS == 1
        assert settings.AUDIO_VAD_ENABLED == True
    
    def test_settings_to_dict(self):
        """Test settings can be exported to dict."""
        from orchestrator.settings import settings
        
        config = settings.to_dict()
        
        assert "audio" in config
        assert "llm" in config
        assert "mqtt" in config
        assert "gpu" in config
    
    def test_gpu_detection(self):
        """Test GPU detection methods."""
        from orchestrator.settings import settings
        
        # Should not raise
        cuda = settings.is_cuda_available()
        cudnn = settings.is_cudnn_available()
        gpu = settings.is_gpu_available()
        
        assert isinstance(cuda, bool)
        assert isinstance(cudnn, bool)
        assert isinstance(gpu, bool)
        
        # GPU requires both CUDA and cuDNN
        if gpu:
            assert cuda and cudnn
