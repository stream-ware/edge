#!/usr/bin/env python3
"""
Log Collector - Zbieranie i analiza logów dla LLM.

Daje LLM dostęp do:
- Logów aplikacji (orchestrator, adaptery)
- Logów systemowych (journalctl)
- Logów Docker
- Błędów i ostrzeżeń

LLM może analizować logi i proaktywnie proponować rozwiązania.
"""

import logging
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import re


@dataclass
class LogEntry:
    """Wpis logu."""
    timestamp: datetime
    level: str
    source: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_str(self) -> str:
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.level} {self.source}: {self.message}"


class LogCollector(logging.Handler):
    """
    Handler logów zbierający wpisy do bufora dla LLM.
    Automatycznie wykrywa błędy i może triggerować proaktywną analizę.
    """
    
    def __init__(self, max_entries: int = 100, on_error: Callable = None):
        super().__init__()
        self.buffer: deque = deque(maxlen=max_entries)
        self.error_buffer: deque = deque(maxlen=20)
        self.warning_buffer: deque = deque(maxlen=20)
        self._on_error_callback = on_error
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        
        self.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
            datefmt='%H:%M:%S'
        ))
    
    def emit(self, record: logging.LogRecord):
        """Przechwytuj wpisy logów."""
        try:
            entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                source=record.name,
                message=record.getMessage(),
                extra={
                    "lineno": record.lineno,
                    "funcName": record.funcName,
                }
            )
            
            self.buffer.append(entry)
            
            if record.levelno >= logging.ERROR:
                self.error_buffer.append(entry)
                if self._on_error_callback:
                    try:
                        self._on_error_callback(entry)
                    except Exception:
                        pass
            elif record.levelno >= logging.WARNING:
                self.warning_buffer.append(entry)
                
        except Exception:
            self.handleError(record)
    
    def get_recent_logs(self, n: int = 20, level: str = None) -> List[LogEntry]:
        """Pobierz ostatnie n wpisów."""
        entries = list(self.buffer)
        if level:
            level_no = getattr(logging, level.upper(), 0)
            entries = [e for e in entries if getattr(logging, e.level, 0) >= level_no]
        return entries[-n:]
    
    def get_errors(self, n: int = 10) -> List[LogEntry]:
        """Pobierz ostatnie błędy."""
        return list(self.error_buffer)[-n:]
    
    def get_warnings(self, n: int = 10) -> List[LogEntry]:
        """Pobierz ostatnie ostrzeżenia."""
        return list(self.warning_buffer)[-n:]
    
    def get_logs_summary(self) -> str:
        """Podsumowanie logów dla LLM."""
        errors = self.get_errors(5)
        warnings = self.get_warnings(5)
        recent = self.get_recent_logs(10)
        
        parts = []
        
        if errors:
            parts.append("BŁĘDY:")
            for e in errors:
                parts.append(f"  - {e.to_str()}")
        
        if warnings:
            parts.append("OSTRZEŻENIA:")
            for w in warnings:
                parts.append(f"  - {w.to_str()}")
        
        if recent:
            parts.append("OSTATNIE LOGI:")
            for r in recent[-5:]:
                parts.append(f"  - {r.to_str()}")
        
        return "\n".join(parts) if parts else "Brak istotnych logów."
    
    def get_context_for_llm(self, include_all: bool = False) -> str:
        """Pełny kontekst logów dla LLM."""
        errors = self.get_errors(10)
        warnings = self.get_warnings(10)
        
        context_parts = []
        
        if errors:
            context_parts.append("=== BŁĘDY W SYSTEMIE ===")
            for e in errors:
                context_parts.append(e.to_str())
        
        if warnings:
            context_parts.append("\n=== OSTRZEŻENIA ===")
            for w in warnings:
                context_parts.append(w.to_str())
        
        if include_all:
            recent = self.get_recent_logs(30)
            context_parts.append("\n=== OSTATNIE LOGI ===")
            for r in recent:
                context_parts.append(r.to_str())
        
        return "\n".join(context_parts) if context_parts else "System działa poprawnie, brak błędów."
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analizuj wzorce w logach."""
        errors = self.get_errors(20)
        
        patterns = {
            "mqtt_issues": [],
            "docker_issues": [],
            "audio_issues": [],
            "camera_issues": [],
            "sensor_issues": [],
            "other_issues": []
        }
        
        for entry in errors:
            msg_lower = entry.message.lower()
            source_lower = entry.source.lower()
            
            if "mqtt" in msg_lower or "mqtt" in source_lower:
                patterns["mqtt_issues"].append(entry)
            elif "docker" in msg_lower or "docker" in source_lower:
                patterns["docker_issues"].append(entry)
            elif "audio" in msg_lower or "stt" in source_lower or "tts" in source_lower:
                patterns["audio_issues"].append(entry)
            elif "camera" in msg_lower or "video" in msg_lower or "vision" in source_lower:
                patterns["camera_issues"].append(entry)
            elif "sensor" in msg_lower:
                patterns["sensor_issues"].append(entry)
            else:
                patterns["other_issues"].append(entry)
        
        return patterns
    
    def get_proactive_suggestions(self) -> List[str]:
        """Generuj proaktywne sugestie na podstawie logów."""
        patterns = self.analyze_patterns()
        suggestions = []
        
        if patterns["mqtt_issues"]:
            suggestions.append("MQTT nie jest połączony. Mogę uruchomić broker komendą 'docker start streamware-mqtt' lub sprawdzić konfigurację.")
        
        if patterns["docker_issues"]:
            suggestions.append("Wykryto problemy z Docker. Mogę sprawdzić status kontenerów lub zrestartować problematyczne serwisy.")
        
        if patterns["camera_issues"]:
            suggestions.append("Kamera nie jest dostępna. Sprawdź czy jest podłączona USB lub zmień indeks w konfiguracji.")
        
        if patterns["audio_issues"]:
            suggestions.append("Wykryto problemy z audio. Sprawdź urządzenia wejścia/wyjścia komendą 'arecord -l' i 'aplay -l'.")
        
        if patterns["sensor_issues"]:
            suggestions.append("Czujniki nie odpowiadają. Sprawdź połączenie I2C lub konfigurację MQTT dla sensorów IoT.")
        
        return suggestions


class ProactiveAnalyzer:
    """
    Analizator proaktywny - LLM analizuje logi i sugeruje rozwiązania.
    """
    
    def __init__(self, log_collector: LogCollector, llm_engine=None):
        self.log_collector = log_collector
        self.llm = llm_engine
        self.logger = logging.getLogger("proactive_analyzer")
        self._last_analysis_time = 0
        self._analysis_interval = 30  # sekundy
        self._pending_suggestions: List[str] = []
    
    async def analyze_and_suggest(self) -> Optional[str]:
        """Analizuj logi i wygeneruj sugestie przez LLM."""
        if not self.llm:
            return None
        
        log_context = self.log_collector.get_context_for_llm()
        patterns = self.log_collector.analyze_patterns()
        
        # Sprawdź czy są istotne problemy
        has_issues = any(len(v) > 0 for v in patterns.values())
        if not has_issues:
            return None
        
        prompt = f"""Przeanalizuj logi systemu i zaproponuj konkretne rozwiązania problemów.

LOGI:
{log_context}

WYKRYTE WZORCE:
- MQTT: {len(patterns['mqtt_issues'])} błędów
- Docker: {len(patterns['docker_issues'])} błędów  
- Audio: {len(patterns['audio_issues'])} błędów
- Kamera: {len(patterns['camera_issues'])} błędów
- Sensory: {len(patterns['sensor_issues'])} błędów

Odpowiedz krótko po polsku:
1. Najważniejszy problem (1 zdanie)
2. Rozwiązanie (1-2 zdania)
3. Komenda naprawcza (jeśli dotyczy)"""

        try:
            response = await self.llm.generate(prompt)
            if response:
                return response.strip()
        except Exception as e:
            self.logger.error(f"LLM analysis error: {e}")
        
        return None
    
    def get_quick_suggestions(self) -> List[str]:
        """Szybkie sugestie bez LLM (na podstawie wzorców)."""
        return self.log_collector.get_proactive_suggestions()
    
    async def should_speak_suggestion(self) -> Optional[str]:
        """Sprawdź czy jest coś ważnego do powiedzenia użytkownikowi."""
        import time
        
        now = time.time()
        if now - self._last_analysis_time < self._analysis_interval:
            return None
        
        self._last_analysis_time = now
        
        # Sprawdź szybkie sugestie
        suggestions = self.get_quick_suggestions()
        if suggestions:
            # Zwróć pierwszą nieogłoszoną sugestię
            for s in suggestions:
                if s not in self._pending_suggestions:
                    self._pending_suggestions.append(s)
                    return s
        
        return None


# Global instance
_log_collector: Optional[LogCollector] = None


def get_log_collector() -> LogCollector:
    """Pobierz globalny kolektor logów."""
    global _log_collector
    if _log_collector is None:
        _log_collector = LogCollector(max_entries=200)
    return _log_collector


def setup_log_collector(on_error: Callable = None) -> LogCollector:
    """Skonfiguruj i zwróć kolektor logów."""
    global _log_collector
    _log_collector = LogCollector(max_entries=200, on_error=on_error)
    
    # Dodaj do root loggera
    root_logger = logging.getLogger()
    root_logger.addHandler(_log_collector)
    
    return _log_collector
