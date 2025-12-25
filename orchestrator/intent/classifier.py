#!/usr/bin/env python3
"""
Intent Classifier - Dynamiczne NLP/LLM rozpoznawanie intencji.

Hierarchia decyzyjna:
1. Domain (docker, sensor, system, diag, env, vision, shell)
2. Intent (restart, status, read, write, help, analyze, etc.)
3. Entities (target, location, value, etc.)

Workflow:
1. Fast pattern match (optymalizacja dla znanych fraz)
2. LLM intent classification (główna ścieżka)
3. Clarification request (jeśli niejasne)
"""

import logging
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Domain(Enum):
    """Główne domeny systemu."""
    DOCKER = "docker"
    SENSOR = "sensor"
    SYSTEM = "system"
    DIAG = "diag"
    ENV = "env"
    VISION = "vision"
    SHELL = "shell"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Rozpoznana intencja."""
    domain: Domain
    action: str
    confidence: float = 1.0
    entities: Dict[str, Any] = field(default_factory=dict)
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    raw_text: str = ""
    source: str = "unknown"


@dataclass
class ConversationContext:
    """Kontekst konwersacji dla lepszego rozumienia."""
    last_domain: Optional[Domain] = None
    last_action: Optional[str] = None
    last_entities: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 10
    
    def add_turn(self, user_text: str, intent: Intent, response: str):
        """Dodaj turę do historii."""
        self.history.append({
            "user": user_text,
            "intent": {"domain": intent.domain.value, "action": intent.action},
            "response": response[:200]
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        self.last_domain = intent.domain
        self.last_action = intent.action
        self.last_entities = intent.entities.copy()
    
    def set_error(self, error: Dict[str, Any]):
        """Zapisz ostatni błąd."""
        self.last_error = error
    
    def get_context_summary(self) -> str:
        """Podsumowanie kontekstu dla LLM."""
        parts = []
        if self.last_domain and self.last_domain != Domain.UNKNOWN:
            parts.append(f"Ostatnia domena: {self.last_domain.value}")
        if self.last_action:
            parts.append(f"Ostatnia akcja: {self.last_action}")
        if self.last_error:
            parts.append(f"Ostatni błąd: {self.last_error.get('error', 'nieznany')}")
        if self.history:
            recent = self.history[-3:]
            history_str = " | ".join([f"U:{h['user'][:30]}" for h in recent])
            parts.append(f"Historia: {history_str}")
        return "; ".join(parts) if parts else "Brak kontekstu"


class IntentClassifier:
    """
    Dynamiczny klasyfikator intencji z hierarchią decyzyjną.
    
    Priorytet:
    1. Explicit commands (krótkie, znane frazy) -> fast path
    2. LLM classification (główna ścieżka)
    3. Context-based inference (użyj kontekstu)
    4. Clarification (poproś o wyjaśnienie)
    """
    
    INTENT_SCHEMA = {
        "docker": {
            "actions": ["restart", "stop", "start", "logs", "status", "list", "inspect"],
            "entities": ["target"],
            "examples": ["zrestartuj backend", "pokaż logi nginx", "status kontenerów"]
        },
        "sensor": {
            "actions": ["read", "write", "list", "subscribe"],
            "entities": ["metric", "location", "value"],
            "examples": ["jaka jest temperatura", "włącz światło w kuchni", "odczytaj wilgotność"]
        },
        "system": {
            "actions": ["help", "exit", "mute", "unmute", "status"],
            "entities": [],
            "examples": ["pomoc", "koniec", "wycisz", "status systemu"]
        },
        "diag": {
            "actions": ["analyze", "check", "fix"],
            "entities": ["topic", "problem"],
            "examples": ["dlaczego nie działa", "zdiagnozuj mqtt", "napraw sieć"]
        },
        "env": {
            "actions": ["get", "set", "unset", "list", "reload"],
            "entities": ["key", "value"],
            "examples": ["pokaż env MQTT_BROKER", "ustaw env DEBUG na true"]
        },
        "vision": {
            "actions": ["detect", "capture", "stream", "list_cameras"],
            "entities": ["camera", "object"],
            "examples": ["co widzisz", "zrób zdjęcie", "wykryj osoby"]
        },
        "shell": {
            "actions": ["run"],
            "entities": ["command"],
            "examples": ["wykonaj komendę docker ps", "uruchom ping localhost"]
        },
        "conversation": {
            "actions": ["greeting", "thanks", "confirm", "deny", "unclear"],
            "entities": [],
            "examples": ["cześć", "dzięki", "tak", "nie", "co?"]
        }
    }
    
    FAST_PATTERNS = [
        (r"^(pomoc|help|co umiesz)\s*[?!.]?$", Domain.SYSTEM, "help", {}),
        (r"^(koniec|exit|wyjdź|quit)\s*[?!.]?$", Domain.SYSTEM, "exit", {}),
        (r"^(cicho|mute|wycisz)\s*[?!.]?$", Domain.SYSTEM, "mute", {}),
        (r"^(odcisz|unmute)\s*[?!.]?$", Domain.SYSTEM, "unmute", {}),
        (r"^(cześć|hej|witaj|hello|hi)\s*[?!.]?$", Domain.CONVERSATION, "greeting", {}),
        (r"^(dzięki|dziękuję|thanks)\s*[?!.]?$", Domain.CONVERSATION, "thanks", {}),
        (r"^(tak|yes|ok|okej)\s*[?!.]?$", Domain.CONVERSATION, "confirm", {}),
        (r"^(nie|no|nope)\s*[?!.]?$", Domain.CONVERSATION, "deny", {}),
    ]
    
    def __init__(self, llm_engine=None, config: dict = None):
        self.logger = logging.getLogger("intent_classifier")
        self.llm = llm_engine
        self.config = config or {}
        self.context = ConversationContext()
        self._classification_prompt = self._build_classification_prompt()
    
    def _build_classification_prompt(self) -> str:
        """Buduje prompt systemowy dla klasyfikacji intencji."""
        schema_desc = []
        for domain, info in self.INTENT_SCHEMA.items():
            actions_str = ", ".join(info["actions"])
            entities_str = ", ".join(info["entities"]) if info["entities"] else "brak"
            examples_str = "; ".join(info["examples"][:2])
            schema_desc.append(f"- {domain}: akcje=[{actions_str}], encje=[{entities_str}], np: {examples_str}")
        
        return f"""Jesteś klasyfikatorem intencji. Analizujesz wypowiedzi użytkownika i zwracasz JSON.

DOMENY I AKCJE:
{chr(10).join(schema_desc)}

ZASADY:
1. Zawsze zwracaj TYLKO JSON (bez markdown, bez tekstu)
2. Jeśli nie jesteś pewien, ustaw confidence < 0.5 i requires_clarification = true
3. Użyj kontekstu konwersacji do lepszego zrozumienia
4. Dla pytań "dlaczego" / "czemu" -> domain=diag, action=analyze
5. Dla niejasnych poleceń -> requires_clarification=true z clarification_prompt

FORMAT ODPOWIEDZI:
{{"domain": "...", "action": "...", "entities": {{}}, "confidence": 0.0-1.0, "requires_clarification": false, "clarification_prompt": null}}"""
    
    async def classify(self, text: str) -> Intent:
        """
        Klasyfikuj intencję użytkownika.
        
        Workflow:
        1. Fast pattern match
        2. LLM classification
        3. Context inference
        4. Clarification
        """
        text = text.strip()
        if not text:
            return Intent(Domain.UNKNOWN, "empty", confidence=0.0, raw_text=text)
        
        text_lower = text.lower()
        
        # 1. Fast pattern match dla znanych fraz
        for pattern, domain, action, entities in self.FAST_PATTERNS:
            if re.match(pattern, text_lower):
                self.logger.debug(f"Fast match: {text} -> {domain.value}.{action}")
                return Intent(domain, action, confidence=1.0, entities=entities, raw_text=text, source="pattern")
        
        # 2. LLM classification
        if self.llm:
            intent = await self._llm_classify(text)
            if intent and intent.confidence >= 0.5:
                return intent
            elif intent and intent.requires_clarification:
                return intent
        
        # 3. Context-based inference
        intent = self._context_inference(text)
        if intent and intent.confidence >= 0.4:
            return intent
        
        # 4. Fallback - ask for clarification
        return Intent(
            Domain.UNKNOWN, "unclear",
            confidence=0.2,
            requires_clarification=True,
            clarification_prompt="Nie rozumiem. Czy chodzi ci o Docker, czujniki, czy coś innego?",
            raw_text=text,
            source="fallback"
        )
    
    async def _llm_classify(self, text: str) -> Optional[Intent]:
        """Klasyfikacja przez LLM."""
        context_summary = self.context.get_context_summary()
        
        prompt = f"""{self._classification_prompt}

KONTEKST: {context_summary}

WYPOWIEDŹ UŻYTKOWNIKA: "{text}"

Odpowiedz TYLKO JSON:"""
        
        try:
            response = await self.llm.generate(prompt)
            if not response:
                return None
            
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"```(?:json)?\n?", "", response)
                response = response.rstrip("`")
            
            data = json.loads(response)
            
            domain_str = data.get("domain", "unknown")
            try:
                domain = Domain(domain_str)
            except ValueError:
                domain = Domain.UNKNOWN
            
            return Intent(
                domain=domain,
                action=data.get("action", "unknown"),
                confidence=float(data.get("confidence", 0.5)),
                entities=data.get("entities", {}),
                requires_clarification=data.get("requires_clarification", False),
                clarification_prompt=data.get("clarification_prompt"),
                raw_text=text,
                source="llm"
            )
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM response not valid JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"LLM classification error: {e}")
            return None
    
    def _context_inference(self, text: str) -> Optional[Intent]:
        """Inferuj intencję z kontekstu."""
        text_lower = text.lower()
        
        # Pytania o błąd
        if any(w in text_lower for w in ["dlaczego", "czemu", "why", "błąd", "error", "nie działa"]):
            if self.context.last_error:
                return Intent(Domain.DIAG, "analyze", confidence=0.7, 
                             entities={"context": "last_error"}, raw_text=text, source="context")
        
        # Kontynuacja poprzedniej domeny
        if self.context.last_domain and self.context.last_domain != Domain.UNKNOWN:
            # Jeśli user mówi "jeszcze raz" / "powtórz"
            if any(w in text_lower for w in ["jeszcze raz", "powtórz", "ponownie", "again"]):
                return Intent(self.context.last_domain, self.context.last_action or "repeat",
                             confidence=0.6, entities=self.context.last_entities, raw_text=text, source="context")
            
            # Jeśli user podaje tylko target/wartość (kontynuacja)
            words = text.split()
            if len(words) <= 2 and self.context.last_action:
                return Intent(self.context.last_domain, self.context.last_action,
                             confidence=0.5, entities={"target": text}, raw_text=text, source="context")
        
        return None
    
    def update_context(self, text: str, intent: Intent, response: str):
        """Aktualizuj kontekst konwersacji."""
        self.context.add_turn(text, intent, response)
    
    def set_error_context(self, error: Dict[str, Any]):
        """Ustaw kontekst błędu."""
        self.context.set_error(error)
    
    def to_dsl(self, intent: Intent) -> Dict[str, Any]:
        """Konwertuj Intent na DSL."""
        if intent.domain == Domain.UNKNOWN or intent.requires_clarification:
            return {
                "action": "system.clarify",
                "prompt": intent.clarification_prompt or "Nie rozumiem, powiedz inaczej.",
                "_source": intent.source,
                "_raw": intent.raw_text,
                "_confidence": intent.confidence
            }
        
        action = f"{intent.domain.value}.{intent.action}"
        
        dsl = {
            "action": action,
            "_source": intent.source,
            "_raw": intent.raw_text,
            "_confidence": intent.confidence
        }
        
        # Map entities to DSL fields
        if intent.entities:
            dsl.update(intent.entities)
        
        return dsl
