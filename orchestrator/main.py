#!/usr/bin/env python3
"""
Streamware Orchestrator - Main Entry Point

LLM-powered orchestrator z:
- Audio interface (STT/TTS)
- MQTT integration
- Docker control
- IoT/Sensor support
- Text2DSL conversion

Optimized for fast startup with lazy loading of heavy modules.
"""

import asyncio
import logging
import signal
import sys
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

import yaml

from .text2dsl import Text2DSL
from .llm_engine import LLMEngine

# Lazy imports for heavy modules (loaded only when needed)
if TYPE_CHECKING:
    from .audio.stt import SpeechToText
    from .audio.tts import TextToSpeech
    from .adapters.docker_adapter import DockerAdapter
    from .adapters.mqtt_adapter import MQTTAdapter
    from .adapters.firmware_adapter import FirmwareAdapter
    from .adapters.env_adapter import EnvAdapter
    from .vision.adapter import VisionAdapter


class Orchestrator:
    """
    Główny orchestrator systemu.
    
    Przepływ:
    1. Audio (STT) / MQTT → Natural Language text
    2. Text2DSL → Structured command (lub LLM fallback)
    3. Adapter → Execute action
    4. Result → Text2DSL → Natural Language
    5. TTS / MQTT → Output
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = logging.getLogger("orchestrator")
        self._load_dotenv()
        self.config = self._load_config(config_path)
        self._apply_env_overrides(self.config)
        
        # Komponenty
        self.text2dsl = Text2DSL(self.config.get("text2dsl", {}))
        self.llm: Optional[LLMEngine] = None
        self.stt: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None
        self.mqtt: Optional[MQTTAdapter] = None
        
        # Adaptery
        self.adapters: Dict[str, Any] = {}
        
        # Stan
        self.running = False
        self.muted = False
    
    def _load_config(self, path: str) -> dict:
        """Wczytanie konfiguracji."""
        config_file = Path(path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        
        # Domyślna konfiguracja
        return {
            "audio": {
                "enabled": True,
                "stt": {"model": "small", "language": "pl"},
                "tts": {"model": "pl_PL-gosia-medium"}
            },
            "llm": {
                "provider": "ollama",
                "model": "phi3:mini"
            },
            "mqtt": {
                "enabled": True,
                "broker": "localhost",
                "port": 1883
            },
            "adapters": {
                "enabled": ["docker", "mqtt", "env"]
            }
        }

    def _load_dotenv(self) -> None:
        env_file = os.environ.get("STREAMWARE_ENV_FILE", ".env")
        try:
            from dotenv import load_dotenv
        except ImportError:
            return

        try:
            p = Path(env_file)
            if p.exists() and p.is_dir():
                env_file = str(p / "streamware.env")
        except Exception:
            pass

        load_dotenv(env_file, override=False)

    def _apply_env_overrides(self, config: dict) -> None:
        mode = os.environ.get("STREAMWARE_MODE")
        if mode:
            config["mode"] = mode

        mqtt_broker = os.environ.get("MQTT_BROKER")
        if mqtt_broker:
            config.setdefault("mqtt", {})["broker"] = mqtt_broker

        mqtt_port = os.environ.get("MQTT_PORT")
        if mqtt_port:
            try:
                config.setdefault("mqtt", {})["port"] = int(mqtt_port)
            except ValueError:
                config.setdefault("mqtt", {})["port"] = mqtt_port

        ollama_host = os.environ.get("OLLAMA_HOST")
        if ollama_host:
            config.setdefault("llm", {})["base_url"] = ollama_host

        prefix = "STREAMWARE__"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            path = [p.lower() for p in key[len(prefix):].split("__") if p]
            if not path:
                continue
            self._set_config_path(config, path, self._coerce_env_value(value))

    def _set_config_path(self, config: dict, path: list, value: Any) -> None:
        node: Any = config
        for part in path[:-1]:
            if not isinstance(node, dict):
                return
            if part not in node or not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]

        if isinstance(node, dict):
            node[path[-1]] = value

    def _coerce_env_value(self, raw: str) -> Any:
        val = raw.strip()
        low = val.lower()
        if low in {"true", "false"}:
            return low == "true"

        try:
            if re.match(r"^-?\d+$", val):
                return int(val)
            if re.match(r"^-?\d+\.\d+$", val):
                return float(val)
        except Exception:
            pass

        if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
            try:
                return json.loads(val)
            except Exception:
                return val

        return val
    
    async def initialize(self):
        """Inicjalizacja komponentów (lazy loading dla szybkiego startu)."""
        self.logger.info("Inicjalizacja Orchestratora...")
        
        # LLM (lightweight, always needed)
        self.logger.info("  → LLM Engine...")
        self.llm = LLMEngine(self.config.get("llm", {}))
        await self.llm.initialize()
        
        # Audio - lazy import (heavy: whisper, piper)
        if self.config.get("audio", {}).get("enabled", True):
            self.logger.info("  → Audio STT (lazy loading)...")
            from .audio.stt import SpeechToText
            self.stt = SpeechToText(
                self.config.get("audio", {}).get("stt", {}),
                self.config.get("audio", {})
            )
            await self.stt.initialize()
            
            self.logger.info("  → Audio TTS (lazy loading)...")
            from .audio.tts import TextToSpeech
            self.tts = TextToSpeech(self.config.get("audio", {}).get("tts", {}))
            await self.tts.initialize()
        
        # MQTT - lazy import
        if self.config.get("mqtt", {}).get("enabled", True):
            self.logger.info("  → MQTT Adapter...")
            from .adapters.mqtt_adapter import MQTTAdapter
            self.mqtt = MQTTAdapter(self.config.get("mqtt", {}))
            await self.mqtt.connect()
            
            # Subscribe na komendy
            await self.mqtt.subscribe("commands/#", self._on_mqtt_command)
            await self.mqtt.subscribe("audio/tts", self._on_mqtt_tts)
        
        # Adaptery - lazy load
        await self._init_adapters()
        
        self.logger.info("✅ Orchestrator zainicjalizowany")
    
    async def _init_adapters(self):
        """Inicjalizacja adapterów (lazy loading)."""
        enabled = self.config.get("adapters", {}).get("enabled", [])
        
        if "docker" in enabled:
            self.logger.info("  → Docker Adapter (lazy loading)...")
            from .adapters.docker_adapter import DockerAdapter
            self.adapters["docker"] = DockerAdapter()
            await self.adapters["docker"].initialize()

        if "env" in enabled:
            self.logger.info("  → Env Adapter (lazy loading)...")
            from .adapters.env_adapter import EnvAdapter
            env_cfg = self.config.get("adapters", {}).get("env", {})
            self.adapters["env"] = EnvAdapter(env_cfg)
            await self.adapters["env"].initialize()
        
        if "firmware" in enabled:
            self.logger.info("  → Firmware Adapter...")
            from .adapters.firmware_adapter import FirmwareAdapter
            self.adapters["firmware"] = FirmwareAdapter(
                self.config.get("firmware", {})
            )
        
        if "vision" in enabled:
            self.logger.info("  → Vision Adapter (lazy loading - YOLO)...")
            from .vision.adapter import VisionAdapter
            self.adapters["vision"] = VisionAdapter(
                self.config.get("vision", {})
            )
            await self.adapters["vision"].initialize()
    
    async def run(self):
        """Główna pętla działania."""
        self.running = True
        
        tasks = []
        
        # Audio loop (jeśli włączone)
        if self.stt:
            tasks.append(asyncio.create_task(
                self._audio_loop(), name="audio"
            ))
        
        # MQTT loop (jeśli włączone)
        if self.mqtt:
            tasks.append(asyncio.create_task(
                self.mqtt.loop(), name="mqtt"
            ))
        
        if not tasks:
            self.logger.warning("Brak aktywnych źródeł input!")
            return
        
        self.logger.info("🎤 Nasłuchuję...")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Taski anulowane")
    
    async def stop(self):
        """Zatrzymanie orchestratora."""
        self.running = False
    
    async def cleanup(self):
        """Zwolnienie zasobów."""
        if self.stt:
            await self.stt.cleanup()
        if self.tts:
            await self.tts.cleanup()
        if self.llm:
            await self.llm.cleanup()
        if self.mqtt:
            await self.mqtt.disconnect()
        
        for adapter in self.adapters.values():
            if hasattr(adapter, 'cleanup'):
                await adapter.cleanup()
    
    # =========================================
    # AUDIO LOOP
    # =========================================
    
    async def _audio_loop(self):
        """Pętla przetwarzania audio."""
        async for transcript in self.stt.stream():
            if not self.running:
                break
            
            if transcript and transcript.strip():
                self.logger.info(f"🎤 STT: {transcript}")
                
                # Publikuj na MQTT (jeśli włączone)
                if self.mqtt:
                    await self.mqtt.publish("audio/stt", transcript)
                
                # Przetwórz komendę
                await self.process_command(transcript, source="audio")
    
    # =========================================
    # MQTT CALLBACKS
    # =========================================
    
    async def _on_mqtt_command(self, topic: str, payload: str):
        """Callback dla komend MQTT."""
        self.logger.info(f"📨 MQTT [{topic}]: {payload}")
        
        # Wyciągnij target z topic (np. commands/backend -> backend)
        parts = topic.split("/")
        target = parts[-1] if len(parts) > 1 else None
        
        await self.process_command(payload, source="mqtt", target_hint=target)
    
    async def _on_mqtt_tts(self, topic: str, payload: str):
        """Callback dla TTS przez MQTT."""
        if self.tts and not self.muted:
            await self.tts.speak(payload)
    
    # =========================================
    # COMMAND PROCESSING
    # =========================================
    
    async def process_command(
        self, 
        text: str, 
        source: str = "unknown",
        target_hint: str = None
    ) -> Optional[str]:
        """
        Główna funkcja przetwarzania komendy.
        
        Args:
            text: Tekst komendy (NL)
            source: Źródło komendy (audio, mqtt, api)
            target_hint: Opcjonalna podpowiedź targetu
            
        Returns:
            Odpowiedź w języku naturalnym
        """
        # 1. Text2DSL - pattern matching
        dsl = self.text2dsl.nl_to_dsl(text)
        
        # 2. LLM fallback jeśli nie rozpoznano
        if not dsl and self.llm:
            self.logger.info("Pattern not matched, using LLM...")
            
            prompt = self.text2dsl.get_llm_prompt(text)
            llm_response = await self.llm.generate(prompt)
            
            if llm_response:
                dsl = self.text2dsl.parse_llm_response(llm_response)
        
        # 3. Jeśli nadal nie ma DSL
        if not dsl:
            response = "Nie rozumiem tej komendy. Powiedz 'pomoc' aby zobaczyć dostępne opcje."
            await self._output_response(response, source)
            return response
        
        # 4. Dodaj target_hint jeśli brak target w DSL
        if target_hint and not dsl.get("target"):
            dsl["target"] = target_hint
        
        self.logger.info(f"📋 DSL: {json.dumps(dsl, ensure_ascii=False)}")
        
        # 5. System commands
        if dsl.get("action", "").startswith("system."):
            result = await self._handle_system_command(dsl)
        else:
            # 6. Execute via adapter
            result = await self._execute_dsl(dsl)
        
        # 7. Convert result to NL
        response = self.text2dsl.dsl_to_nl(result)
        
        self.logger.info(f"💬 Response: {response}")
        
        # 8. Output
        await self._output_response(response, source, dsl)
        
        return response
    
    async def _handle_system_command(self, dsl: dict) -> dict:
        """Obsługa komend systemowych."""
        action = dsl.get("action")
        
        if action == "system.exit":
            await self.stop()
            return {"action": action, "status": "ok"}
        
        elif action == "system.mute":
            self.muted = True
            return {"action": action, "status": "ok"}
        
        elif action == "system.help":
            return {"action": action, "status": "ok"}
        
        return {"action": action, "status": "unknown"}
    
    async def _execute_dsl(self, dsl: dict) -> dict:
        """Wykonanie DSL przez odpowiedni adapter."""
        action = dsl.get("action", "")
        
        # Wyciągnij kategorię z akcji (np. docker.restart -> docker)
        category = action.split(".")[0] if "." in action else action
        
        # Znajdź adapter
        adapter = self.adapters.get(category)
        
        if not adapter:
            return {
                "action": action,
                "status": "error",
                "error": f"Adapter '{category}' nie jest dostępny"
            }
        
        try:
            result = await adapter.execute(dsl)
            result["action"] = action
            return result
        except Exception as e:
            self.logger.exception(f"Adapter error: {e}")
            return {
                "action": action,
                "status": "error",
                "error": str(e)
            }
    
    async def _output_response(
        self, 
        response: str, 
        source: str,
        dsl: dict = None
    ):
        """Wysłanie odpowiedzi do odpowiedniego kanału."""
        # TTS (jeśli źródło audio i nie wyciszony)
        if source == "audio" and self.tts and not self.muted:
            await self.tts.speak(response)
        
        # MQTT
        if self.mqtt:
            target = dsl.get("target", "system") if dsl else "system"
            
            # Event z wynikiem
            event_data = {
                "response": response,
                "dsl": dsl,
                "source": source
            }
            await self.mqtt.publish(f"events/{target}", json.dumps(event_data))


def setup_logging(level: str = "INFO"):
    """Konfiguracja logowania."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
        datefmt="%H:%M:%S"
    )


def print_banner():
    """Banner startowy."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🎯 STREAMWARE ORCHESTRATOR                              ║
    ║   LLM-powered Docker/IoT Controller                       ║
    ║                                                           ║
    ║   Softreck / prototypowanie.pl                            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


async def main():
    """Main entry point."""
    print_banner()
    setup_logging("INFO")
    
    logger = logging.getLogger("main")
    
    # Config path (z argumentu lub domyślny)
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    
    orchestrator = Orchestrator(config_path)
    
    # Signal handlers
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        logger.info("Otrzymano sygnał stop...")
        asyncio.create_task(orchestrator.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        await orchestrator.initialize()
        
        print("\n" + "=" * 60)
        print("🎤 Nasłuchuję... (Ctrl+C = stop)")
        print("   Powiedz 'pomoc' aby zobaczyć dostępne komendy")
        print("=" * 60 + "\n")
        
        await orchestrator.run()
        
    except Exception as e:
        logger.exception(f"Błąd: {e}")
    finally:
        await orchestrator.cleanup()
        logger.info("Orchestrator zakończony.")


if __name__ == "__main__":
    asyncio.run(main())
