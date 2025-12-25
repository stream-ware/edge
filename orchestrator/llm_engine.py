"""
LLM Engine - Wrapper dla lokalnych modeli LLM

Obsługuje:
- Ollama (domyślnie)
- llama.cpp
- Fallback do API (opcjonalnie)
"""

import asyncio
import logging
from typing import Optional
import json

try:
    import httpx
except ImportError:
    httpx = None


class LLMEngine:
    """
    Wrapper dla LLM inference.
    
    Używany jako fallback gdy Text2DSL nie rozpozna komendy.
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger("llm")
        
        self.provider = config.get("provider", "ollama")
        self.model = config.get("model", "phi3:mini")
        self.base_url = config.get("base_url", "http://localhost:11434")
        
        self.temperature = config.get("temperature", 0.3)  # Niższa dla DSL
        self.max_tokens = config.get("max_tokens", 256)
        self.timeout = config.get("timeout", 30.0)
        
        self.system_prompt = config.get("system_prompt", self._default_system_prompt())
        
        self._client: Optional[httpx.AsyncClient] = None
    
    def _default_system_prompt(self) -> str:
        return """Jesteś asystentem konwertującym komendy użytkownika na JSON DSL.
Odpowiadaj TYLKO poprawnym JSON bez dodatkowego tekstu.
Nie dodawaj wyjaśnień, tylko sam JSON."""
    
    async def initialize(self):
        """Inicjalizacja klienta HTTP."""
        if httpx is None:
            raise ImportError("httpx not installed")
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
        
        # Test połączenia
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                self.logger.info(f"✅ Połączono z Ollama ({self.model})")
            else:
                self.logger.warning(f"Ollama status: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Nie można połączyć z Ollama: {e}")
    
    async def cleanup(self):
        """Zamknięcie klienta."""
        if self._client:
            await self._client.aclose()
    
    async def generate(self, prompt: str) -> Optional[str]:
        """
        Generacja odpowiedzi LLM.
        
        Args:
            prompt: Prompt do LLM
            
        Returns:
            Odpowiedź LLM lub None
        """
        if not self._client:
            return None
        
        try:
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = await self._client.post("/api/chat", json=body)
            
            if response.status_code != 200:
                self.logger.error(f"Ollama error: {response.status_code}")
                return None
            
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            
            return content
            
        except httpx.TimeoutException:
            self.logger.error("LLM timeout")
            return None
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return None
    
    async def generate_nl_response(
        self, 
        user_input: str, 
        context: dict = None
    ) -> Optional[str]:
        """
        Generacja odpowiedzi w języku naturalnym.
        
        Używane gdy potrzeba bardziej rozbudowanej odpowiedzi.
        """
        context_str = json.dumps(context, ensure_ascii=False) if context else ""
        
        prompt = f"""Odpowiedz użytkownikowi krótko i pomocnie po polsku.

Kontekst: {context_str}
Użytkownik: {user_input}

Odpowiedź:"""
        
        # Zmień system prompt dla NL
        original_system = self.system_prompt
        self.system_prompt = "Jesteś pomocnym asystentem. Odpowiadaj krótko i konkretnie po polsku."
        
        response = await self.generate(prompt)
        
        self.system_prompt = original_system
        return response
