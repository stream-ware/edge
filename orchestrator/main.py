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
import time
import difflib
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

import yaml

from .text2dsl import Text2DSL
from .llm_engine import LLMEngine
from .intent import IntentClassifier, Intent, Domain
from .log_collector import setup_log_collector, ProactiveAnalyzer, LogCollector

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
        
        # Log collector (zbiera logi dla LLM)
        self.log_collector: LogCollector = setup_log_collector(on_error=self._on_log_error)
        self.proactive_analyzer: Optional[ProactiveAnalyzer] = None
        
        # Komponenty
        self.text2dsl = Text2DSL(self.config.get("text2dsl", {}))
        self.intent_classifier: Optional[IntentClassifier] = None
        self.llm: Optional[LLMEngine] = None
        self.stt: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None
        self.mqtt: Optional[MQTTAdapter] = None
        
        # Adaptery
        self.adapters: Dict[str, Any] = {}
        
        # Stan
        self.running = False
        self.muted = False
        self._tts_task: Optional[asyncio.Task] = None
        self._last_tts_text: Optional[str] = None
        self._last_tts_time: float = 0.0
        self._last_error: Optional[Dict[str, Any]] = None
        self._last_command: Optional[str] = None

        barge_in_cfg = (self.config.get("audio", {}) or {}).get("barge_in", {}) or {}
        self._barge_in_enabled = bool(barge_in_cfg.get("enabled", True))
        self._barge_in_trigger = str(barge_in_cfg.get("trigger", "transcript") or "transcript").strip().lower()
        self._echo_suppression_enabled = bool(barge_in_cfg.get("echo_suppression", True))
    
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

            if key.endswith("__INPUT_DEVICE") and not value.strip():
                coerced_value = None
            else:
                coerced_value = self._coerce_env_value(value)

            self._set_config_path(config, path, coerced_value)

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
        
        # Intent Classifier (uses LLM)
        self.logger.info("  → Intent Classifier...")
        self.intent_classifier = IntentClassifier(
            llm_engine=self.llm,
            config=self.config.get("intent", {})
        )
        
        # Proactive Analyzer (LLM + logi)
        self.logger.info("  → Proactive Analyzer...")
        self.proactive_analyzer = ProactiveAnalyzer(self.log_collector, self.llm)
        
        # Audio - lazy import (heavy: whisper, piper)
        if self.config.get("audio", {}).get("enabled", True):
            self.logger.info("  → Audio STT (lazy loading)...")
            from .audio.stt import SpeechToText

            on_speech_start = self._on_speech_start if (self._barge_in_enabled and self._barge_in_trigger == "vad") else None
            self.stt = SpeechToText(
                self.config.get("audio", {}).get("stt", {}),
                self.config.get("audio", {}),
                on_speech_start=on_speech_start
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
            # NOTE: do not initialize on startup; initialize on first vision command
        
        self.logger.info("  → Shell Adapter (diagnostics)...")
        from .adapters.shell_adapter import ShellAdapter
        shell_cfg = self.config.get("adapters", {}).get("shell", {})
        self.adapters["shell"] = ShellAdapter(shell_cfg)
        await self.adapters["shell"].initialize()
    
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

        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
        self._tts_task = None
    
    async def cleanup(self):
        """Zwolnienie zasobów."""
        if self.stt:
            await self.stt.cleanup()
        if self.tts:
            try:
                await self.tts.stop()
            except Exception:
                pass
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
                if self.tts and self.tts.is_speaking:
                    if self._is_echo_transcript(transcript):
                        continue

                    if self._barge_in_enabled and self._barge_in_trigger == "transcript":
                        self._stop_tts_playback()

                self.logger.info(f"🎤 STT: {transcript}")
                
                # Publikuj na MQTT (jeśli włączone)
                if self.mqtt:
                    await self.mqtt.publish("audio/stt", transcript)
                
                # Przetwórz komendę
                await self.process_command(transcript, source="audio")
                
                # Sprawdź proaktywne sugestie po przetworzeniu komendy
                await self._check_proactive_suggestions()
    
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
            self._start_tts(payload)

    def _on_speech_start(self):
        if not self._barge_in_enabled or self._barge_in_trigger != "vad":
            return
        self._stop_tts_playback()

    def _stop_tts_playback(self) -> None:
        if self.tts and self.tts.is_speaking:
            asyncio.create_task(self.tts.stop())

        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
        self._tts_task = None

    def _norm_text(self, text: str) -> str:
        t = (text or "").lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^a-z0-9ąćęłńóśźż ]+", "", t)
        return t.strip()

    def _is_echo_transcript(self, transcript: str) -> bool:
        if not self._echo_suppression_enabled:
            return False
        if not self._last_tts_text:
            return False
        if (time.time() - self._last_tts_time) > 10.0:
            return False

        a = self._norm_text(transcript)
        b = self._norm_text(self._last_tts_text)
        if not a or not b:
            return False

        if a in b or b in a:
            return True

        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        return ratio >= 0.78
    
    def _on_log_error(self, entry):
        """Callback wywoływany przy każdym błędzie w logach."""
        # Można tu dodać natychmiastowe reakcje na błędy
        pass
    
    async def _check_proactive_suggestions(self):
        """Sprawdź czy LLM ma coś do powiedzenia na podstawie logów."""
        if not self.proactive_analyzer:
            return
        
        suggestion = await self.proactive_analyzer.should_speak_suggestion()
        if suggestion and self.tts and not self.muted:
            self.logger.info(f"💡 Proaktywna sugestia: {suggestion[:100]}...")
            self._start_tts(f"Zauważyłem problem. {suggestion}")
    
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
        cmd_id = str(uuid.uuid4())[:8]
        started_at = time.perf_counter()
        self._last_command = text
        text = (text or "").strip()
        self.logger.info(f"🧭 cmd[{cmd_id}] input source={source} text={text!r}")

        # 1. Fast path: Text2DSL pattern matching (unika LLM dla prostych komend)
        intent: Optional[Intent] = None
        dsl = self.text2dsl.nl_to_dsl(text)
        if dsl:
            self.logger.info(
                f"🧭 cmd[{cmd_id}] decision route=text2dsl action={dsl.get('action')} source={dsl.get('_source')}"
            )

        # 2. Intent classification (LLM) tylko jeśli nie rozpoznano wzorcem
        if not dsl and self.intent_classifier:
            intent = await self.intent_classifier.classify(text)
            self.logger.debug(
                f"Intent: {intent.domain.value}.{intent.action} (conf={intent.confidence:.2f}, src={intent.source})"
            )
            dsl = self.intent_classifier.to_dsl(intent)
            if dsl:
                self.logger.info(
                    f"🧭 cmd[{cmd_id}] decision route=intent action={dsl.get('action')} source={dsl.get('_source')} confidence={dsl.get('_confidence')}"
                )

        # 3. LLM fallback dla text2dsl (ostatnia deska ratunku)
        if not dsl and self.llm:
            self.logger.info("Using LLM DSL fallback...")
            prompt = self.text2dsl.get_llm_prompt(text)
            llm_response = await self.llm.generate(prompt)
            if llm_response:
                dsl = self.text2dsl.parse_llm_response(llm_response)
                if dsl:
                    self.logger.info(
                        f"🧭 cmd[{cmd_id}] decision route=llm_dsl action={dsl.get('action')} source={dsl.get('_source')}"
                    )

        # 4. Jeśli nadal nie ma DSL -> proaktywna prośba o doprecyzowanie
        if not dsl:
            dsl = {
                "action": "system.clarify",
                "prompt": "Nie jestem pewien o co chodzi. Co chcesz zrobić?",
                "options": [
                    "status systemu",
                    "temperatura",
                    "status kontenerów Docker",
                    "zdiagnozuj problemy",
                    "pomoc"
                ],
                "_source": "fallback",
                "_raw": text,
                "_confidence": 0.2
            }
        
        # 5. Dodaj target_hint jeśli brak target w DSL
        if target_hint and not dsl.get("target"):
            dsl["target"] = target_hint

        # 5b. Disambiguacja temperatury (otoczenie vs CPU/GPU vs IoT)
        if (dsl.get("action") == "sensor.read" and (dsl.get("metric") or "") == "temperature"):
            text_lower = text.lower()
            has_explicit_scope = any(
                k in text_lower
                for k in [
                    "cpu", "gpu", "procesor", "karta", "otoczenia", "pokój", "salon", "kuchnia",
                    "sypialnia", "iot", "smart", "smarthome", "mqtt", "komputera"
                ]
            )
            if not has_explicit_scope and (dsl.get("location") in {None, "default"}):
                dsl = {
                    "action": "system.clarify",
                    "prompt": "O jaką temperaturę chodzi?",
                    "options": [
                        "temperatura komputera (CPU/GPU)",
                        "temperatura otoczenia (czujnik pokojowy)",
                        "temperatura urządzenia IoT/SmartHome",
                        "temperatura z MQTT"
                    ],
                    "_source": "disambiguation",
                    "_raw": text,
                    "_confidence": 0.7,
                    "_suggested_domain": "sensor",
                    "_suggested_action": "read",
                    "_entities": {"metric": "temperature"}
                }
        
        self.logger.info(f"📋 DSL: {json.dumps(dsl, ensure_ascii=False)}")
        
        # 6. Route to appropriate handler
        action = dsl.get("action") or ""

        # Guard: never execute shell.run unless explicitly requested.
        if action == "shell.run":
            src = str(dsl.get("_source") or "")
            cmd = (dsl.get("command") or "").strip()
            explicit = bool(re.search(r"\b(wykonaj|uruchom)\b.*\b(komend|command|shell)\b", text, flags=re.IGNORECASE))
            if (not explicit) and src in {"llm", "context", "fallback", "proactive"}:
                self.logger.warning(f"🧭 cmd[{cmd_id}] blocked shell.run from source={src} cmd={cmd!r}")
                dsl = {
                    "action": "system.clarify",
                    "prompt": "Wygląda na to, że chcesz wykonać komendę w shell. Czy mam ją uruchomić?",
                    "options": [
                        f"tak: wykonaj komendę {cmd}" if cmd else "tak: wykonaj komendę ...",
                        "nie",
                        "pokaż logi",
                        "zdiagnozuj system"
                    ],
                    "_source": "guard",
                    "_raw": text,
                    "_confidence": 0.9
                }
                action = dsl.get("action")
        
        if action == "system.clarify":
            response = self._format_clarification(dsl)
            self.logger.info(f"💬 Response: {response}")
            await self._output_response(response, source, dsl)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            self.logger.info(f"🧭 cmd[{cmd_id}] done clarify elapsed_ms={elapsed_ms}")
            return response
        elif action.startswith("system."):
            result = await self._handle_system_command(dsl)
        elif action.startswith("diag.") or action == "shell.run":
            result = await self._handle_diagnostic(dsl)
        elif action.startswith("conversation."):
            result = await self._handle_conversation(dsl)
        else:
            # Execute via adapter
            result = await self._execute_dsl(dsl)
        
        # Track errors for diagnostic analysis
        if result.get("status") == "error":
            self._last_error = {
                "command": text,
                "dsl": dsl,
                "result": result,
                "time": time.time()
            }
            if self.intent_classifier:
                self.intent_classifier.set_error_context(self._last_error)
        
        # 7. Convert result to NL
        response = self.text2dsl.dsl_to_nl(result)
        
        self.logger.info(f"💬 Response: {response}")
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        self.logger.info(f"🧭 cmd[{cmd_id}] done action={action} status={result.get('status')} elapsed_ms={elapsed_ms}")
        
        # 8. Update conversation context
        if self.intent_classifier:
            if intent is None and isinstance(action, str) and action:
                try:
                    prefix = action.split(".")[0] if "." in action else action
                    verb = action.split(".", 1)[1] if "." in action else "unknown"
                    domain = Domain(prefix) if prefix in {d.value for d in Domain} else Domain.UNKNOWN
                    intent = Intent(domain, verb, confidence=float(dsl.get("_confidence", 1.0) or 1.0), raw_text=text, source=str(dsl.get("_source", "pattern")))
                except Exception:
                    intent = None

            if intent:
                self.intent_classifier.update_context(text, intent, response)
        
        # 9. Output
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
            if self.tts:
                try:
                    await self.tts.stop()
                except Exception:
                    pass
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
            self._tts_task = None
            return {"action": action, "status": "ok"}

        elif action == "system.unmute":
            self.muted = False
            return {"action": action, "status": "ok"}

        elif action == "system.tts.stop":
            if self.tts:
                try:
                    await self.tts.stop()
                except Exception:
                    pass
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
            self._tts_task = None
            return {"action": action, "status": "ok"}
        
        elif action == "system.help":
            return {"action": action, "status": "ok"}
        
        return {"action": action, "status": "unknown"}
    
    async def _handle_conversation(self, dsl: dict) -> dict:
        """Obsługa intencji konwersacyjnych (powitania, podziękowania, etc.)."""
        action = dsl.get("action", "")
        
        responses = {
            "conversation.greeting": "Cześć! Jak mogę pomóc?",
            "conversation.thanks": "Nie ma za co!",
            "conversation.confirm": "OK, rozumiem.",
            "conversation.deny": "Rozumiem, anuluję.",
            "conversation.unclear": "Nie rozumiem. Powiedz 'pomoc' aby zobaczyć opcje.",
        }
        
        response_text = responses.get(action, "Słucham?")
        return {"action": action, "status": "ok", "response": response_text}
    
    async def _handle_diagnostic(self, dsl: dict) -> dict:
        """Obsługa komend diagnostycznych z workflow: informacja → analiza → rozwiązanie."""
        action = dsl.get("action", "")
        shell = self.adapters.get("shell")
        
        if action == "shell.run":
            command = dsl.get("command", "")
            if not command:
                return {"action": action, "status": "error", "error": "Brak komendy"}
            if shell:
                result = await shell.execute(command)
                result["action"] = action
                return result
            return {"action": action, "status": "error", "error": "Shell adapter niedostępny"}
        
        elif action == "diag.check":
            topic = dsl.get("topic", "system")
            if shell:
                diag_result = await shell.diagnose(topic)
                summary_parts = []
                for d in diag_result.get("diagnostics", []):
                    if d.get("status") == "ok" and d.get("stdout"):
                        summary_parts.append(d["stdout"][:200])
                    elif d.get("status") == "error":
                        summary_parts.append(f"❌ {d.get('command', '')}: {d.get('stderr', d.get('error', ''))[:100]}")
                
                summary = "\n".join(summary_parts[:8]) if summary_parts else f"Brak danych dla {topic}"
                
                if self.llm:
                    analysis_prompt = f"""Przeanalizuj wyniki diagnostyki '{topic}' i podaj krótkie podsumowanie (1-2 zdania) co działa, a co nie:

{summary}

Odpowiedz po polsku, krótko i konkretnie."""
                    llm_summary = await self.llm.generate(analysis_prompt)
                    if llm_summary:
                        summary = llm_summary.strip()
                
                return {"action": action, "status": "ok", "topic": topic, "summary": summary}
            return {"action": action, "status": "error", "error": "Shell adapter niedostępny"}
        
        elif action == "diag.logs":
            # Pokaż logi z kontekstem LLM
            log_context = self.log_collector.get_context_for_llm(include_all=True)
            
            if self.llm:
                prompt = f"""Przeanalizuj logi systemu i powiedz co się dzieje. Skup się na błędach i ostrzeżeniach.

{log_context}

Odpowiedz po polsku:
1. Główne problemy (jeśli są)
2. Co działa poprawnie
3. Sugestie naprawy (jeśli potrzebne)"""
                
                analysis = await self.llm.generate(prompt)
                if analysis:
                    return {"action": action, "status": "ok", "analysis": analysis.strip()}
            
            return {"action": action, "status": "ok", "analysis": log_context[:500]}
        
        elif action == "diag.analyze":
            # Pobierz kontekst logów dla analizy
            log_context = self.log_collector.get_context_for_llm()
            
            if not self._last_error and not log_context.strip():
                return {"action": action, "status": "ok", "analysis": "System działa poprawnie, brak błędów."}
            
            error_info = self._last_error or {}
            error_msg = error_info.get("result", {}).get("error", "")
            command = error_info.get("command", "")
            dsl_info = error_info.get("dsl", {})
            
            context_parts = []
            if command:
                context_parts.append(f"Ostatnia komenda: {command}")
            if error_msg:
                context_parts.append(f"Błąd: {error_msg}")
            
            # Dodaj kontekst logów
            context_parts.append(f"\nLOGI SYSTEMU:\n{log_context}")
            
            if shell:
                action_type = dsl_info.get("action", "").split(".")[0]
                topic_map = {"sensor": "sensors", "docker": "docker", "mqtt": "mqtt", "env": "system"}
                topic = topic_map.get(action_type, "system")
                diag_result = await shell.diagnose(topic)
                for d in diag_result.get("diagnostics", [])[:3]:
                    if d.get("stdout"):
                        context_parts.append(d["stdout"][:150])
            
            if self.llm:
                analysis_prompt = f"""Użytkownik wykonał komendę która się nie powiodła. Przeanalizuj problem i zaproponuj rozwiązanie.

{chr(10).join(context_parts)}

Odpowiedz po polsku:
1. Krótka analiza przyczyny (1 zdanie)
2. Konkretne rozwiązanie (1-2 zdania)"""
                
                llm_response = await self.llm.generate(analysis_prompt)
                if llm_response:
                    return {"action": action, "status": "ok", "analysis": llm_response.strip()}
            
            return {"action": action, "status": "ok", "analysis": f"Ostatni błąd: {error_msg}. Sprawdź konfigurację lub połączenie."}
        
        elif action == "diag.fix":
            problem = dsl.get("problem", "")
            
            if self.llm:
                fix_prompt = f"""Użytkownik chce naprawić problem: "{problem}"

Zaproponuj konkretne kroki naprawy (max 3 kroki). Jeśli to wymaga komendy shell, podaj ją.
Odpowiedz po polsku, krótko i konkretnie."""
                
                llm_response = await self.llm.generate(fix_prompt)
                if llm_response:
                    return {"action": action, "status": "ok", "result": llm_response.strip()}
            
            return {"action": action, "status": "ok", "result": f"Nie mogę automatycznie naprawić: {problem}. Spróbuj 'zdiagnozuj system'."}
        
        return {"action": action, "status": "error", "error": f"Nieznana akcja diagnostyczna: {action}"}
    
    async def _execute_dsl(self, dsl: dict) -> dict:
        """Wykonanie DSL przez odpowiedni adapter."""
        action = dsl.get("action") or ""

        if not isinstance(action, str):
            action = str(action)

        if not action.strip():
            return {
                "action": None,
                "status": "error",
                "error": "Brak pola 'action' w DSL"
            }
        
        # Wyciągnij kategorię z akcji (np. docker.restart -> docker)
        category = action.split(".")[0] if "." in action else action
        
        # Znajdź adapter
        adapter = self.adapters.get(category)
        if adapter is None and category in {"sensor", "device"}:
            # sensor.* / device.* obsługuje FirmwareAdapter
            adapter = self.adapters.get("firmware")
        
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
        if source == "audio" and self.tts and not self.muted:
            action = (dsl.get("action") if dsl else None)
            if action != "system.tts.stop":
                self._start_tts(self._tts_text(response))
        
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

    def _format_clarification(self, dsl: dict) -> str:
        """Formatuj odpowiedź z opcjami do wyboru."""
        prompt = dsl.get("prompt", "Powiedz więcej...")
        options = dsl.get("options", [])
        
        if not options:
            return prompt
        
        # Format dla TTS (krótsza wersja)
        if len(options) <= 3:
            options_str = ", ".join(options[:3])
            return f"{prompt} Na przykład: {options_str}"
        else:
            # Dla wielu opcji - tylko 3 pierwsze w TTS
            options_str = ", ".join(options[:3])
            return f"{prompt} Na przykład: {options_str}, lub inne."
    
    def _tts_text(self, text: str) -> str:
        if not text:
            return text
        one_line = " ".join([t.strip() for t in text.splitlines() if t.strip()])
        if len(one_line) <= 240:
            return one_line
        return one_line[:240] + "..."

    def _start_tts(self, text: str) -> None:
        if not self.tts or not text:
            return

        self._last_tts_text = text
        self._last_tts_time = time.time()

        if self._tts_task and not self._tts_task.done():
            asyncio.create_task(self.tts.stop())
            self._tts_task.cancel()
            self._tts_task = None

        self._tts_task = asyncio.create_task(self.tts.speak(text))


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
