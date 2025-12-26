"""
Text2DSL - Natural Language to Domain Specific Language converter

Konwertuje komendy w języku naturalnym na strukturyzowane DSL (JSON)
i z powrotem na odpowiedzi w języku naturalnym.

Obsługuje:
- Polskie i angielskie komendy
- Docker operations
- IoT/Sensor operations  
- SQL queries
- Custom actions
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class DSLCommand:
    """Reprezentacja komendy DSL."""
    action: str
    target: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    raw_text: str = ""
    confidence: float = 1.0


class Text2DSL:
    """
    Konwerter Natural Language <-> DSL.
    
    Używa:
    1. Pattern matching (szybkie, deterministyczne)
    2. LLM fallback (dla nieznanych komend)
    """
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("text2dsl")
        self.config = config or {}
        
        # Polskie i angielskie patterns
        self._init_patterns()
        
        # Cache dla LLM-generated DSL
        self._cache: Dict[str, dict] = {}
    
    def _init_patterns(self):
        """Inicjalizacja wzorców rozpoznawania."""

        self.env_patterns = [
            (r"(pokaż|odczytaj|get)\s+(zmienn[aę]\s+)?(env|środowiskow[aą])\s+([A-Za-z_][A-Za-z0-9_]*)",
             lambda m: {"action": "env.get", "key": m.group(4)}),

            (r"(ustaw|set)\s+(zmienn[aę]\s+)?(env|środowiskow[aą])\s+([A-Za-z_][A-Za-z0-9_]*)\s+(na|to)\s+(.+)",
             lambda m: {"action": "env.set", "key": m.group(4), "value": m.group(6).strip()}),

            (r"(usuń|unset)\s+(zmienn[aę]\s+)?(env|środowiskow[aą])\s+([A-Za-z_][A-Za-z0-9_]*)",
             lambda m: {"action": "env.unset", "key": m.group(4)}),

            (r"(lista|pokaż)\s+(zmiennych|zmienne)\s+(env|środowiskowe)",
             lambda m: {"action": "env.list"}),

            (r"(edytuj|otwórz)\s+(\.env|env)",
             lambda m: {"action": "env.editor"}),

            (r"(przeładuj|reload)\s+(\.env|env)",
             lambda m: {"action": "env.reload"}),
        ]
        
        # Docker patterns
        self.docker_patterns = [
            # Restart
            (r"(zrestartuj|restart|uruchom ponownie)\s+(\w+)", 
             lambda m: {"action": "docker.restart", "target": m.group(2)}),
            
            # Stop
            (r"(zatrzymaj|stop|wyłącz)\s+(\w+)",
             lambda m: {"action": "docker.stop", "target": m.group(2)}),
            
            # Start
            (r"(uruchom|start|włącz)\s+(\w+)",
             lambda m: {"action": "docker.start", "target": m.group(2)}),
            
            # Logs
            (r"(pokaż|wyświetl|show)\s+(logi|logs)\s+(\w+)",
             lambda m: {"action": "docker.logs", "target": m.group(3), "tail": 10}),
            
            (r"(logi|logs)\s+(\w+)\s+(\d+)\s+(linii|lines)",
             lambda m: {"action": "docker.logs", "target": m.group(2), "tail": int(m.group(3))}),
            
            # Status
            (r"(status|stan)\s+(kontenerów|containers|wszystkich)",
             lambda m: {"action": "docker.status"}),
            
            (r"(status|stan)\s+(\w+)",
             lambda m: {"action": "docker.inspect", "target": m.group(2)}),
            
            # List
            (r"(lista|list|pokaż)\s+(kontenerów|containers)",
             lambda m: {"action": "docker.list"}),
        ]
        
        # IoT/Sensor patterns
        self.sensor_patterns = [
            # Temperature
            (r"(jaka jest|podaj|odczytaj|sprawdź|zmierz)?\s*(temperatur\w*|temperature)\s*(w|in)?\s*(\w+)?",
             lambda m: {"action": "sensor.read", "metric": "temperature",
                       "location": m.group(4) or "default"}),
            
            # Humidity
            (r"(jaka jest |podaj )?(wilgotność|humidity)\s*(w |in )?(\w+)?",
             lambda m: {"action": "sensor.read", "metric": "humidity",
                       "location": m.group(4) or "default"}),
            
            # Generic sensor
            (r"odczytaj\s+(\w+)\s+(z |from )?(\w+)",
             lambda m: {"action": "sensor.read", "metric": m.group(1),
                       "device": m.group(3)}),
            
            # Device control
            (r"(włącz|turn on|zapal)\s+(światło|light)\s*(w |in )?(\w+)?",
             lambda m: {"action": "device.set", "device": "light",
                       "location": m.group(4) or "default", "state": "on"}),
            
            (r"(wyłącz|turn off|zgaś)\s+(światło|light)\s*(w |in )?(\w+)?",
             lambda m: {"action": "device.set", "device": "light",
                       "location": m.group(4) or "default", "state": "off"}),
            
            # Set value
            (r"ustaw\s+(\w+)\s+na\s+(\d+)",
             lambda m: {"action": "device.set", "device": m.group(1),
                       "value": int(m.group(2))}),
        ]
        
        # SQL patterns
        self.sql_patterns = [
            (r"(zapytaj|query)\s+(bazę|database)\s+o\s+(.+)",
             lambda m: {"action": "sql.query", "query_hint": m.group(3)}),
            
            (r"(ile|how many)\s+(\w+)\s+w\s+bazie",
             lambda m: {"action": "sql.count", "table": m.group(2)}),
            
            (r"(pokaż|show)\s+(\w+)\s+z\s+bazy",
             lambda m: {"action": "sql.select", "table": m.group(2)}),
        ]
        
        # Vision patterns
        self.vision_patterns = [
            (r"^(zobacz|spójrz|patrz|look)[!?.]*$",
             lambda m: {"action": "system.clarify", "prompt": "Co mam sprawdzić?", "options": ["co widzisz", "ile osób widzisz", "lista kamer", "dodaj kamerę RTSP"]}),
            # Describe/What do you see
            (r"(co widzisz|co jest na|opisz|describe|what do you see)",
             lambda m: {"action": "vision.describe"}),
            
            (r"(co widzisz|co jest)\s+(na|przez|w)\s+(\w+)",
             lambda m: {"action": "vision.describe", "camera": m.group(3)}),
            
            # Count objects
            (r"(ile|how many)\s+(\w+)\s+(widzisz|jest|are)",
             lambda m: {"action": "vision.count", "target": m.group(2)}),
            
            # Find object
            (r"(gdzie jest|znajdź|find|where is)\s+(\w+)",
             lambda m: {"action": "vision.find", "target": m.group(2)}),
            
            # Add camera
            (r"dodaj kamerę\s+(rtsp://\S+)",
             lambda m: {"action": "vision.add_camera", "source": m.group(1)}),
            
            (r"dodaj kamerę\s+(\d+)",
             lambda m: {"action": "vision.add_camera", "source": int(m.group(1))}),
            
            (r"(podłącz|connect)\s+(kamerę|camera)\s+(\S+)",
             lambda m: {"action": "vision.add_camera", "source": m.group(3)}),
            
            # Remove camera
            (r"(usuń|odłącz|remove)\s+(kamerę|camera)\s+(\w+)",
             lambda m: {"action": "vision.remove_camera", "name": m.group(3)}),
            
            # List cameras
            (r"(lista|list|pokaż)\s+(kamer|cameras)",
             lambda m: {"action": "vision.list_cameras"}),
            
            # Scan network for RTSP
            (r"(skanuj|scan)\s+(sieć|network)\s*(rtsp)?",
             lambda m: {"action": "vision.scan_network"}),
            
            (r"(znajdź|find)\s+(kamery|cameras)\s+(w sieci|on network)",
             lambda m: {"action": "vision.scan_network"}),
        ]
        
        # Diagnostic patterns
        self.diagnostic_patterns = [
            (r"^(dlaczego|why|czemu)[?!.]*$",
             lambda m: {"action": "diag.analyze", "context": "last_error"}),
            
            (r"(dlaczego|why|czemu)\s+(nie działa|not working|nie udało|failed|błąd|error|to|tak)",
             lambda m: {"action": "diag.analyze", "context": "last_error"}),
            
            (r"(zdiagnozuj|diagnose|sprawdź)\s+(mqtt|docker|sieć|network|sensory|sensors|audio|kamerę|camera|system)",
             lambda m: {"action": "diag.check", "topic": m.group(2).lower().replace("sieć", "network").replace("sensory", "sensors").replace("kamerę", "camera")}),
            
            (r"(napraw|fix|rozwiąż|solve)\s+(.+)",
             lambda m: {"action": "diag.fix", "problem": m.group(2)}),
            
            (r"(wykonaj|run|uruchom)\s+(komendę|command|shell)\s+(.+)",
             lambda m: {"action": "shell.run", "command": m.group(3)}),
            
            (r"(co jest|what is)\s+(nie tak|wrong|zepsute|broken)",
             lambda m: {"action": "diag.analyze", "context": "last_error"}),
            
            (r"(status|stan)\s+(systemu|system)",
             lambda m: {"action": "diag.check", "topic": "system"}),

            (r"(pokaż|show|analizuj|analyze)\s+(logi|logs)\s*(systemu|system)?",
             lambda m: {"action": "diag.logs"}),
            
            (r"(co się dzieje|what.s happening|status logów)",
             lambda m: {"action": "diag.logs"}),
        ]
        
        # System patterns
        self.system_patterns = [
            (r"(pomoc|help|co umiesz)",
             lambda m: {"action": "system.help"}),
            
            (r"(koniec|exit|wyjdź|quit)",
             lambda m: {"action": "system.exit"}),
            
            (r"(cicho|mute|wycisz)",
             lambda m: {"action": "system.mute"}),

            (r"(odcisz|unmute|mów|wznów głos)",
             lambda m: {"action": "system.unmute"}),

            (r"(zatrzymaj tts|stop tts|przerwij mówienie|przerwij głos)",
             lambda m: {"action": "system.tts.stop"}),
        ]
    
    def nl_to_dsl(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Konwertuje tekst w języku naturalnym na DSL.
        
        Args:
            text: Komenda w języku naturalnym
            
        Returns:
            Dict z akcją DSL lub None jeśli nie rozpoznano
        """
        if not text:
            return None
        
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        
        # Sprawdź cache
        if text_lower in self._cache:
            self.logger.debug(f"Cache hit: {text_lower}")
            return self._cache[text_lower]
        
        # Próbuj pattern matching
        # env.* musi zachować wielkość liter kluczy i wartości (np. MQTT_BROKER)
        for pattern, handler in self.env_patterns:
            match = re.search(pattern, text_stripped, flags=re.IGNORECASE)
            if match:
                dsl = handler(match)
                dsl["_source"] = "pattern"
                dsl["_raw"] = text
                self.logger.info(f"Pattern match: '{text}' -> {dsl}")
                self._cache[text_lower] = dsl
                return dsl

        # Kolejność ma znaczenie: komendy IoT/urządzeń (np. "włącz światło")
        # nie mogą zostać błędnie zinterpretowane jako docker.start.
        all_patterns = (
            self.diagnostic_patterns +
            self.sensor_patterns +
            self.docker_patterns +
            self.sql_patterns +
            self.vision_patterns +
            self.system_patterns
        )
        
        for pattern, handler in all_patterns:
            match = re.search(pattern, text_lower)
            if match:
                dsl = handler(match)
                dsl["_source"] = "pattern"
                dsl["_raw"] = text
                
                self.logger.info(f"Pattern match: '{text}' -> {dsl}")
                self._cache[text_lower] = dsl
                return dsl
        
        # Nie znaleziono - zwróć None (LLM fallback w orchestratorze)
        self.logger.warning(f"No pattern match for: '{text}'")
        return None
    
    def dsl_to_nl(self, dsl: Dict[str, Any]) -> str:
        """
        Konwertuje wynik DSL na tekst w języku naturalnym.
        
        Args:
            dsl: Wynik wykonania akcji DSL
            
        Returns:
            Odpowiedź w języku naturalnym
        """
        if not dsl:
            return "Nie wykonano żadnej akcji."
        
        action = dsl.get("action", "")
        status = dsl.get("status", "unknown")
        target = dsl.get("target", "")
        
        # Docker responses
        if action == "docker.restart":
            if status == "ok":
                return f"Kontener {target} został zrestartowany pomyślnie."
            else:
                error = dsl.get("error", "nieznany błąd")
                return f"Nie udało się zrestartować kontenera {target}: {error}"
        
        elif action == "docker.stop":
            if status == "ok":
                return f"Kontener {target} został zatrzymany."
            else:
                return f"Nie udało się zatrzymać kontenera {target}."
        
        elif action == "docker.start":
            if status == "ok":
                return f"Kontener {target} został uruchomiony."
            else:
                return f"Nie udało się uruchomić kontenera {target}."
        
        elif action == "docker.logs":
            logs = dsl.get("logs", "")
            if logs:
                # Skróć jeśli za długie
                if len(logs) > 500:
                    logs = logs[:500] + "..."
                return f"Ostatnie logi z {target}:\n{logs}"
            else:
                return f"Brak logów z kontenera {target}."
        
        elif action == "docker.status":
            containers = dsl.get("containers", [])
            if containers:
                running = [c["name"] for c in containers if c.get("status") == "running"]
                stopped = [c["name"] for c in containers if c.get("status") != "running"]
                
                msg = f"Masz {len(containers)} kontenerów."
                if running:
                    msg += f" Uruchomione: {', '.join(running)}."
                if stopped:
                    msg += f" Zatrzymane: {', '.join(stopped)}."
                return msg
            else:
                return "Nie znaleziono żadnych kontenerów."
        
        elif action == "docker.list":
            containers = dsl.get("containers", [])
            if containers:
                names = [c.get("name", "?") for c in containers]
                return f"Kontenery: {', '.join(names)}."
            return "Brak kontenerów."
        
        # Sensor responses
        elif action == "sensor.read":
            metric = dsl.get("metric", "wartość")
            value = dsl.get("value")
            unit = dsl.get("unit", "")
            location = dsl.get("location", "")
            
            if value is not None:
                loc_str = f" w {location}" if location and location != "default" else ""
                return f"{metric.capitalize()}{loc_str} wynosi {value} {unit}."
            else:
                return f"Nie udało się odczytać {metric}."
        
        elif action == "device.set":
            device = dsl.get("device", "urządzenie")
            state = dsl.get("state", dsl.get("value", "?"))
            location = dsl.get("location", "")
            
            if status == "ok":
                loc_str = f" w {location}" if location and location != "default" else ""
                return f"{device.capitalize()}{loc_str} ustawione na {state}."
            else:
                return f"Nie udało się ustawić {device}."

        elif action == "env.get":
            if status == "ok":
                return f"{dsl.get('key')}={dsl.get('value')}"
            return f"Nie udało się odczytać zmiennej {dsl.get('key', '')}."

        elif action == "env.set":
            if status == "ok":
                return f"Ustawiono {dsl.get('key')} w .env."
            return f"Nie udało się ustawić zmiennej {dsl.get('key', '')}."

        elif action == "env.unset":
            if status == "ok":
                return f"Usunięto {dsl.get('key')} z .env."
            return f"Nie udało się usunąć zmiennej {dsl.get('key', '')}."

        elif action == "env.list":
            if status == "ok":
                keys = dsl.get("keys", [])
                if not keys:
                    return "Brak zmiennych w .env."
                preview = ", ".join(keys[:20])
                if len(keys) > 20:
                    preview += ", ..."
                return f"Zmiennych w .env: {len(keys)}. Klucze: {preview}"
            return "Nie udało się pobrać listy zmiennych .env."

        elif action == "env.editor":
            if status == "ok":
                return "Otworzyłem edytor pliku .env."
            return "Nie udało się otworzyć edytora .env."

        elif action == "env.reload":
            if status == "ok":
                return f"Przeładowano zmienne z .env ({dsl.get('count', 0)})."
            return "Nie udało się przeładować .env."
        
        # Vision responses
        elif action == "vision.describe":
            description = dsl.get("description", "Brak opisu")
            return description
        
        elif action == "vision.detect":
            count = dsl.get("count", 0)
            description = dsl.get("description", "")
            if count > 0:
                return f"Wykryto {count} obiektów. {description}"
            else:
                return "Nie wykryto żadnych obiektów."
        
        elif action == "vision.count":
            target = dsl.get("target", "obiektów")
            count = dsl.get("count", 0)
            return dsl.get("message", f"Widzę {count} {target}")
        
        elif action == "vision.find":
            found = dsl.get("found", False)
            message = dsl.get("message", "")
            return message
        
        elif action == "vision.add_camera":
            camera = dsl.get("camera", "kamera")
            if status == "ok":
                return f"Kamera {camera} została dodana."
            else:
                return f"Nie udało się dodać kamery {camera}."
        
        elif action == "vision.remove_camera":
            camera = dsl.get("camera", "kamera")
            return f"Kamera {camera} została usunięta."
        
        elif action == "vision.list_cameras":
            cameras = dsl.get("cameras", [])
            if cameras:
                names = [c.get("name", "?") for c in cameras]
                connected = sum(1 for c in cameras if c.get("connected"))
                return f"Masz {len(cameras)} kamer ({connected} połączonych): {', '.join(names)}."
            return "Brak skonfigurowanych kamer."
        
        elif action == "vision.scan_network":
            count = dsl.get("count", 0)
            cameras = dsl.get("cameras", [])
            if count > 0:
                return f"Znaleziono {count} kamer RTSP w sieci: {', '.join(cameras[:5])}"
            return "Nie znaleziono kamer RTSP w sieci."
        
        # SQL responses
        elif action == "sql.query":
            results = dsl.get("results", [])
            count = dsl.get("count", len(results))
            return f"Zapytanie zwróciło {count} wyników."
        
        elif action == "sql.count":
            count = dsl.get("count", 0)
            table = dsl.get("table", "rekordów")
            return f"W tabeli {table} jest {count} rekordów."
        
        # System responses
        elif action == "system.help":
            return self._get_help_text()
        
        elif action == "system.exit":
            return "Do zobaczenia!"
        
        elif action == "system.mute":
            return "Wyciszam."

        elif action == "system.unmute":
            return "Odciszam."

        elif action == "system.tts.stop":
            return "Zatrzymuję."
        
        # Diagnostic responses
        elif action == "diag.analyze":
            analysis = dsl.get("analysis", "")
            suggestion = dsl.get("suggestion", "")
            if analysis and suggestion:
                return f"{analysis} {suggestion}"
            elif analysis:
                return analysis
            return "Analizuję problem..."
        
        elif action == "diag.check":
            topic = dsl.get("topic", "system")
            summary = dsl.get("summary", "")
            if summary:
                return summary
            return f"Sprawdzam {topic}..."
        
        elif action == "diag.fix":
            result = dsl.get("result", "")
            if result:
                return result
            return "Próbuję naprawić..."
        
        elif action == "diag.logs":
            analysis = dsl.get("analysis", "")
            if analysis:
                return analysis
            return "Analizuję logi..."
        
        elif action == "shell.run":
            cmd = dsl.get("command", "")
            rc = dsl.get("returncode", None)
            duration_ms = dsl.get("duration_ms", None)

            if status == "rejected":
                return f"Odrzucono komendę: {cmd or '[brak]'} ({dsl.get('error', 'niedozwolona')})"

            if status == "timeout":
                return f"Timeout podczas wykonywania: {cmd or '[brak]'}"

            if status == "error":
                err = dsl.get("stderr") or dsl.get("error") or "nieznany błąd"
                return f"Błąd przy wykonywaniu: {cmd or '[brak]'} ({err})"

            output = dsl.get("stdout", "")
            header_parts = []
            if cmd:
                header_parts.append(f"Komenda: {cmd}")
            if rc is not None:
                header_parts.append(f"rc={rc}")
            if duration_ms is not None:
                header_parts.append(f"{duration_ms}ms")

            header = " (" + ", ".join(header_parts[1:]) + ")" if len(header_parts) > 1 else ""
            prefix = header_parts[0] + header + "\n" if header_parts else ""

            if output:
                lines = output.split('\n')
                if len(lines) > 8:
                    return prefix + '\n'.join(lines[:8]) + f"\n... ({len(lines)} linii)"
                return prefix + output

            return prefix + "Wykonano."
        
        # Conversation responses
        elif action and action.startswith("conversation."):
            return dsl.get("response", "Słucham?")
        
        # Generic
        if status == "ok":
            return "Akcja wykonana pomyślnie."
        elif status == "error":
            error = dsl.get("error", "nieznany błąd")
            return f"Wystąpił błąd: {error}"
        
        return "Akcja zakończona."
    
    def _get_help_text(self) -> str:
        """Tekst pomocy."""
        return """Mogę ci pomóc z:
        
Docker:
- "Zrestartuj backend" - restart kontenera
- "Pokaż logi frontend" - wyświetl logi
- "Status kontenerów" - lista i stan

Czujniki:
- "Jaka jest temperatura" - odczyt temperatury
- "Włącz światło w kuchni" - sterowanie

Diagnostyka:
- "Dlaczego nie działa?" - analiza ostatniego błędu
- "Zdiagnozuj mqtt/docker/sieć" - sprawdź komponenty
- "Napraw X" - próba automatycznej naprawy

System:
- "Pomoc" - ta wiadomość
- "Koniec" - zakończenie

ENV (.env):
- "Pokaż env MQTT_BROKER" - odczyt zmiennej
- "Ustaw env MQTT_BROKER na mqtt" - zapis do .env"""
    
    def parse_llm_response(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """
        Parsuje odpowiedź LLM jako DSL.
        
        LLM może zwrócić JSON bezpośrednio lub tekst z JSON.
        """
        if not llm_response:
            return None
        
        # Szukaj JSON w odpowiedzi
        json_match = re.search(r'\{[^{}]+\}', llm_response)
        
        if json_match:
            try:
                dsl = json.loads(json_match.group())
                dsl["_source"] = "llm"
                return dsl
            except json.JSONDecodeError:
                pass
        
        return None
    
    def get_llm_prompt(self, user_text: str) -> str:
        """
        Generuje prompt dla LLM do konwersji NL -> DSL.
        """
        return f"""Przekonwertuj poniższą komendę użytkownika na JSON DSL.

Dostępne akcje:
- docker.restart, docker.stop, docker.start, docker.logs, docker.status, docker.list
- sensor.read, device.set
- env.get, env.set, env.unset, env.list, env.editor, env.reload
- sql.query, sql.count, sql.select
- system.help, system.exit

Komenda użytkownika: "{user_text}"

Odpowiedz TYLKO poprawnym JSON bez dodatkowego tekstu.
Przykład: {{"action": "docker.restart", "target": "backend"}}

JSON:"""
