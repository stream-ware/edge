"""
Streamware Settings - Global Configuration from Environment

Usage:
    from orchestrator.settings import settings
    
    print(settings.AUDIO_STT_DEVICE)
    print(settings.MQTT_BROKER)
    print(settings.is_gpu_available())
"""

import os
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache


def _load_dotenv():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = os.environ.get("STREAMWARE_ENV_FILE", ".env")
    try:
        p = Path(env_file)
        if p.exists() and p.is_dir():
            env_file = str(p / "streamware.env")
    except Exception:
        pass

    load_dotenv(env_file, override=False)


_load_dotenv()


def _get_env(key: str, default: Any = None, cast: type = str) -> Any:
    """Get environment variable with type casting."""
    value = os.environ.get(key)
    if value is None:
        return default
    
    if cast == bool:
        return value.lower() in ("true", "1", "yes", "on")
    
    try:
        return cast(value)
    except (ValueError, TypeError):
        return default


def _get_list(key: str, default: list = None, separator: str = ",") -> list:
    """Get environment variable as list."""
    value = os.environ.get(key)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(separator) if item.strip()]


class Settings:
    """
    Global settings loaded from environment variables.
    
    Naming convention:
    - STREAMWARE__ prefix for all nested config
    - Double underscore (__) separates nested levels
    - Example: STREAMWARE__AUDIO__STT__MODEL -> audio.stt.model
    """
    
    # ============================================
    # GENERAL
    # ============================================
    @property
    def MODE(self) -> str:
        return _get_env("STREAMWARE_MODE", "docker")
    
    @property
    def LOG_LEVEL(self) -> str:
        return _get_env("STREAMWARE_LOG_LEVEL", "INFO")
    
    # ============================================
    # AUDIO
    # ============================================
    @property
    def AUDIO_ENABLED(self) -> bool:
        return _get_env("STREAMWARE__AUDIO__ENABLED", True, bool)
    
    @property
    def AUDIO_SAMPLE_RATE(self) -> int:
        return _get_env("STREAMWARE__AUDIO__SAMPLE_RATE", 16000, int)
    
    @property
    def AUDIO_CHANNELS(self) -> int:
        return _get_env("STREAMWARE__AUDIO__CHANNELS", 1, int)
    
    @property
    def AUDIO_INPUT_DEVICE(self) -> Optional[str]:
        val = _get_env("STREAMWARE__AUDIO__INPUT_DEVICE", "")
        if isinstance(val, str):
            val = val.strip()

        if not val:
            return None

        if isinstance(val, str) and val.lower() == "auto":
            return None
        try:
            return int(val)
        except ValueError:
            return val
    
    # STT
    @property
    def AUDIO_STT_MODEL(self) -> str:
        return _get_env("STREAMWARE__AUDIO__STT__MODEL", "small")
    
    @property
    def AUDIO_STT_LANGUAGE(self) -> str:
        return _get_env("STREAMWARE__AUDIO__STT__LANGUAGE", "pl")
    
    @property
    def AUDIO_STT_DEVICE(self) -> str:
        return _get_env("STREAMWARE__AUDIO__STT__DEVICE", "auto")
    
    @property
    def AUDIO_STT_COMPUTE_TYPE(self) -> str:
        return _get_env("STREAMWARE__AUDIO__STT__COMPUTE_TYPE", "auto")
    
    # VAD
    @property
    def AUDIO_VAD_ENABLED(self) -> bool:
        return _get_env("STREAMWARE__AUDIO__VAD__ENABLED", True, bool)
    
    @property
    def AUDIO_VAD_MODE(self) -> int:
        return _get_env("STREAMWARE__AUDIO__VAD__MODE", 3, int)
    
    @property
    def AUDIO_VAD_SILENCE_DURATION(self) -> float:
        return _get_env("STREAMWARE__AUDIO__VAD__SILENCE_DURATION", 0.8, float)
    
    @property
    def AUDIO_VAD_MAX_BUFFER_SECONDS(self) -> float:
        return _get_env("STREAMWARE__AUDIO__VAD__MAX_BUFFER_SECONDS", 10.0, float)
    
    # TTS
    @property
    def AUDIO_TTS_MODEL(self) -> str:
        return _get_env("STREAMWARE__AUDIO__TTS__MODEL", "pl_PL-gosia-medium")
    
    @property
    def AUDIO_TTS_LENGTH_SCALE(self) -> float:
        return _get_env("STREAMWARE__AUDIO__TTS__LENGTH_SCALE", 1.0, float)
    
    @property
    def AUDIO_TTS_SAMPLE_RATE(self) -> int:
        return _get_env("STREAMWARE__AUDIO__TTS__SAMPLE_RATE", 22050, int)
    
    # ============================================
    # LLM
    # ============================================
    @property
    def LLM_PROVIDER(self) -> str:
        return _get_env("STREAMWARE__LLM__PROVIDER", "ollama")
    
    @property
    def LLM_MODEL(self) -> str:
        return _get_env("STREAMWARE__LLM__MODEL", "phi3:mini")
    
    @property
    def LLM_BASE_URL(self) -> str:
        return _get_env("STREAMWARE__LLM__BASE_URL", 
                       _get_env("OLLAMA_HOST", "http://localhost:11434"))
    
    @property
    def LLM_TEMPERATURE(self) -> float:
        return _get_env("STREAMWARE__LLM__TEMPERATURE", 0.3, float)
    
    @property
    def LLM_MAX_TOKENS(self) -> int:
        return _get_env("STREAMWARE__LLM__MAX_TOKENS", 256, int)
    
    @property
    def LLM_TIMEOUT(self) -> float:
        return _get_env("STREAMWARE__LLM__TIMEOUT", 30.0, float)
    
    # ============================================
    # MQTT
    # ============================================
    @property
    def MQTT_ENABLED(self) -> bool:
        return _get_env("STREAMWARE__MQTT__ENABLED", True, bool)
    
    @property
    def MQTT_BROKER(self) -> str:
        return _get_env("STREAMWARE__MQTT__BROKER",
                       _get_env("MQTT_BROKER", "localhost"))
    
    @property
    def MQTT_PORT(self) -> int:
        return _get_env("STREAMWARE__MQTT__PORT",
                       _get_env("MQTT_PORT", 1883, int), int)
    
    @property
    def MQTT_USERNAME(self) -> Optional[str]:
        return _get_env("STREAMWARE__MQTT__USERNAME", None)
    
    @property
    def MQTT_PASSWORD(self) -> Optional[str]:
        return _get_env("STREAMWARE__MQTT__PASSWORD", None)
    
    @property
    def MQTT_CLIENT_ID(self) -> str:
        return _get_env("STREAMWARE__MQTT__CLIENT_ID", "streamware-orchestrator")
    
    # ============================================
    # DOCKER
    # ============================================
    @property
    def DOCKER_SOCKET(self) -> str:
        return _get_env("STREAMWARE__ADAPTERS__DOCKER__SOCKET", 
                       "unix:///var/run/docker.sock")
    
    @property
    def DOCKER_TIMEOUT(self) -> int:
        return _get_env("STREAMWARE__ADAPTERS__DOCKER__TIMEOUT", 30, int)
    
    # ============================================
    # VISION
    # ============================================
    @property
    def VISION_MODEL(self) -> str:
        return _get_env("STREAMWARE__VISION__MODEL", "yolov8n")
    
    @property
    def VISION_CONFIDENCE(self) -> float:
        return _get_env("STREAMWARE__VISION__CONFIDENCE", 0.5, float)
    
    @property
    def VISION_IOU_THRESHOLD(self) -> float:
        return _get_env("STREAMWARE__VISION__IOU_THRESHOLD", 0.45, float)
    
    @property
    def VISION_MAX_DETECTIONS(self) -> int:
        return _get_env("STREAMWARE__VISION__MAX_DETECTIONS", 20, int)
    
    # ============================================
    # ADAPTERS
    # ============================================
    @property
    def ADAPTERS_ENABLED(self) -> list:
        return _get_list("STREAMWARE__ADAPTERS__ENABLED", 
                        ["docker", "mqtt", "firmware", "vision", "env"])
    
    @property
    def FIRMWARE_SIMULATION(self) -> bool:
        return _get_env("STREAMWARE__ADAPTERS__FIRMWARE__SIMULATION", True, bool)
    
    # ============================================
    # EMBEDDED
    # ============================================
    @property
    def EMBEDDED_POWER_MODE(self) -> int:
        return _get_env("STREAMWARE__EMBEDDED__POWER_MODE", 0, int)
    
    @property
    def EMBEDDED_GPU_MEMORY_FRACTION(self) -> float:
        return _get_env("STREAMWARE__EMBEDDED__GPU_MEMORY_FRACTION", 0.8, float)
    
    @property
    def EMBEDDED_USE_TENSORRT(self) -> bool:
        return _get_env("STREAMWARE__EMBEDDED__USE_TENSORRT", True, bool)
    
    # ============================================
    # DOCKER COMPOSE PORTS
    # ============================================
    @property
    def WEBAPP_PORT(self) -> int:
        return _get_env("WEBAPP_PORT", 8080, int)
    
    @property
    def MQTT_PORT_EXTERNAL(self) -> int:
        return _get_env("MQTT_PORT_EXTERNAL", 1883, int)
    
    @property
    def MOSQUITTO_WS_PORT(self) -> int:
        return _get_env("MOSQUITTO_WS_PORT", 9001, int)
    
    # ============================================
    # GPU DETECTION
    # ============================================
    @lru_cache(maxsize=1)
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available via torch or ctranslate2."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import ctranslate2
            return "cuda" in ctranslate2.get_supported_compute_types()
        except (ImportError, Exception):
            pass
        
        return False
    
    @lru_cache(maxsize=1)
    def is_cudnn_available(self) -> bool:
        """Check if cuDNN libraries are accessible."""
        import ctypes
        import ctypes.util

        mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)

        def _can_load_any(candidates: list[str]) -> bool:
            for cand in candidates:
                if not cand:
                    continue
                try:
                    ctypes.CDLL(cand, mode=mode)
                    return True
                except OSError:
                    continue
            return False

        # IMPORTANT:
        # CTranslate2 (used by faster-whisper) loads cuDNN by SONAME at runtime.
        # If the dynamic linker can't resolve these names, it can abort the process.
        # Therefore we only return True when BOTH core + ops libs are loadable by name.
        core_candidates = [
            ctypes.util.find_library("cudnn"),
            "libcudnn.so.9",
            "libcudnn.so",
        ]

        ops_candidates = [
            ctypes.util.find_library("cudnn_ops"),
            "libcudnn_ops.so.9.1.0",
            "libcudnn_ops.so.9.1",
            "libcudnn_ops.so.9",
            "libcudnn_ops.so",
        ]

        core_ok = _can_load_any(core_candidates)
        ops_ok = _can_load_any(ops_candidates)

        if core_ok and ops_ok:
            return True

        allow_pip = _get_env("STREAMWARE__GPU__ALLOW_PIP_CUDNN", False, bool)
        if allow_pip and self._try_preload_cudnn_from_pip(mode=mode):
            core_ok = _can_load_any(core_candidates)
            ops_ok = _can_load_any(ops_candidates)
            return core_ok and ops_ok

        return False
    
    def _try_preload_cudnn_from_pip(self, mode: int = 0) -> bool:
        """Try to preload cuDNN from nvidia-cudnn-cu12 pip package."""
        import glob
        
        try:
            import nvidia.cudnn
        except Exception:
            return False
        
        file_attr = getattr(nvidia.cudnn, "__file__", None)
        if file_attr:
            pkg_dir = os.path.dirname(file_attr)
        else:
            path_attr = getattr(nvidia.cudnn, "__path__", None)
            if path_attr:
                pkg_dir = list(path_attr)[0] if path_attr else ""
            else:
                return False
        
        if not pkg_dir:
            return False
        
        lib_dir = os.path.join(pkg_dir, "lib")
        if not os.path.isdir(lib_dir):
            return False
        
        import ctypes

        core_candidates = glob.glob(os.path.join(lib_dir, "libcudnn.so*") )
        ops_candidates = glob.glob(os.path.join(lib_dir, "libcudnn_ops.so*") )
        extra_candidates = glob.glob(os.path.join(lib_dir, "libcudnn_*.so*") )

        core_loaded = False
        ops_loaded = False

        if not mode:
            mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)

        for path in sorted(set(core_candidates)):
            try:
                ctypes.CDLL(path, mode=mode)
                core_loaded = True
                break
            except OSError:
                continue

        for path in sorted(set(ops_candidates)):
            try:
                ctypes.CDLL(path, mode=mode)
                ops_loaded = True
                break
            except OSError:
                continue

        for path in sorted(set(extra_candidates)):
            try:
                ctypes.CDLL(path, mode=mode)
            except OSError:
                continue

        return core_loaded and ops_loaded
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is fully available (CUDA + cuDNN)."""
        return self.is_cuda_available() and self.is_cudnn_available()
    
    def get_effective_device(self) -> str:
        """Get effective device based on config and availability."""
        device = self.AUDIO_STT_DEVICE
        if device == "auto":
            return "cuda" if self.is_gpu_available() else "cpu"
        return device
    
    def get_effective_compute_type(self) -> str:
        """Get effective compute type based on config and device."""
        compute_type = self.AUDIO_STT_COMPUTE_TYPE
        if compute_type == "auto":
            return "float16" if self.get_effective_device() == "cuda" else "float32"
        return compute_type
    
    def to_dict(self) -> dict:
        """Export all settings as dictionary (for debugging)."""
        return {
            "mode": self.MODE,
            "log_level": self.LOG_LEVEL,
            "audio": {
                "enabled": self.AUDIO_ENABLED,
                "sample_rate": self.AUDIO_SAMPLE_RATE,
                "channels": self.AUDIO_CHANNELS,
                "input_device": self.AUDIO_INPUT_DEVICE,
                "stt": {
                    "model": self.AUDIO_STT_MODEL,
                    "language": self.AUDIO_STT_LANGUAGE,
                    "device": self.get_effective_device(),
                    "compute_type": self.get_effective_compute_type(),
                },
                "vad": {
                    "enabled": self.AUDIO_VAD_ENABLED,
                    "mode": self.AUDIO_VAD_MODE,
                    "silence_duration": self.AUDIO_VAD_SILENCE_DURATION,
                    "max_buffer_seconds": self.AUDIO_VAD_MAX_BUFFER_SECONDS,
                },
                "tts": {
                    "model": self.AUDIO_TTS_MODEL,
                    "length_scale": self.AUDIO_TTS_LENGTH_SCALE,
                    "sample_rate": self.AUDIO_TTS_SAMPLE_RATE,
                },
            },
            "llm": {
                "provider": self.LLM_PROVIDER,
                "model": self.LLM_MODEL,
                "base_url": self.LLM_BASE_URL,
                "temperature": self.LLM_TEMPERATURE,
                "max_tokens": self.LLM_MAX_TOKENS,
                "timeout": self.LLM_TIMEOUT,
            },
            "mqtt": {
                "enabled": self.MQTT_ENABLED,
                "broker": self.MQTT_BROKER,
                "port": self.MQTT_PORT,
                "client_id": self.MQTT_CLIENT_ID,
            },
            "adapters": {
                "enabled": self.ADAPTERS_ENABLED,
            },
            "vision": {
                "model": self.VISION_MODEL,
                "confidence": self.VISION_CONFIDENCE,
            },
            "gpu": {
                "cuda_available": self.is_cuda_available(),
                "cudnn_available": self.is_cudnn_available(),
                "gpu_available": self.is_gpu_available(),
            },
        }


# Global singleton instance
settings = Settings()
