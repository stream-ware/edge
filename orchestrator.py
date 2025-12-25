"""
Orchestrator - Główna logika koordynująca komponenty

Zarządza przepływem danych między:
- Audio input (mikrofon) → STT
- Video input (kamera) → Vision  
- STT + Vision → LLM
- LLM → TTS → Audio output
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque
import time

from src.audio.stt import SpeechToText
from src.audio.tts import TextToSpeech
from src.vision.detector import ObjectDetector
from src.llm.inference import LLMInference


@dataclass
class DetectedObject:
    """Wykryty obiekt z kamery."""
    label: str
    confidence: float
    position: str  # "lewo-góra", "środek", etc.
    bbox: tuple  # (x1, y1, x2, y2)


@dataclass
class ConversationContext:
    """Kontekst rozmowy."""
    last_objects: List[DetectedObject] = field(default_factory=list)
    last_speech: str = ""
    last_response: str = ""
    timestamp: float = 0.0


class Orchestrator:
    """
    Główny koordynator systemu.
    
    Przepływ:
    1. Audio stream → VAD → STT (gdy wykryto mowę)
    2. Video stream → YOLO (co N klatek)
    3. Gdy STT zwróci tekst:
       - Pobierz ostatnie wykrycia z Vision
       - Zbuduj prompt dla LLM
       - Wyślij do LLM
       - Odpowiedź → TTS
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("orchestrator")
        
        # Komponenty
        self.stt: Optional[SpeechToText] = None
        self.tts: Optional[TextToSpeech] = None
        self.detector: Optional[ObjectDetector] = None
        self.llm: Optional[LLMInference] = None
        
        # Stan
        self.running = False
        self.context = ConversationContext()
        
        # Kolejki komunikacji
        self.speech_queue: asyncio.Queue = asyncio.Queue()
        self.vision_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        
        # Bufor obiektów (ostatnie wykrycia)
        self._detected_objects: deque = deque(maxlen=30)
        
        # Konfiguracja orchestratora
        orch_config = config.get("orchestrator", {})
        self.mode = orch_config.get("mode", "continuous")
        self.idle_timeout = orch_config.get("idle_timeout", 300)
        
        # Bufor (opcjonalny)
        buffer_config = orch_config.get("buffer", {})
        self.buffer_enabled = buffer_config.get("enabled", False)
    
    async def initialize(self):
        """Inicjalizacja wszystkich komponentów."""
        self.logger.info("Inicjalizacja STT...")
        self.stt = SpeechToText(self.config.get("stt", {}), self.config.get("audio", {}))
        await self.stt.initialize()
        
        self.logger.info("Inicjalizacja TTS...")
        self.tts = TextToSpeech(self.config.get("tts", {}))
        await self.tts.initialize()
        
        self.logger.info("Inicjalizacja Vision...")
        self.detector = ObjectDetector(self.config.get("vision", {}))
        await self.detector.initialize()
        
        self.logger.info("Inicjalizacja LLM...")
        self.llm = LLMInference(self.config.get("llm", {}))
        await self.llm.initialize()
        
        self.logger.info("✅ Wszystkie komponenty zainicjalizowane")
    
    async def run(self):
        """Główna pętla działania."""
        self.running = True
        
        # Uruchomienie tasków
        tasks = [
            asyncio.create_task(self._audio_loop(), name="audio"),
            asyncio.create_task(self._vision_loop(), name="vision"),
            asyncio.create_task(self._processing_loop(), name="processing"),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Taski anulowane")
    
    async def stop(self):
        """Zatrzymanie systemu."""
        self.logger.info("Zatrzymywanie...")
        self.running = False
    
    async def cleanup(self):
        """Czyszczenie zasobów."""
        if self.stt:
            await self.stt.cleanup()
        if self.tts:
            await self.tts.cleanup()
        if self.detector:
            await self.detector.cleanup()
        if self.llm:
            await self.llm.cleanup()
    
    # =========================================
    # AUDIO LOOP
    # =========================================
    
    async def _audio_loop(self):
        """Pętla przetwarzania audio."""
        self.logger.info("🎤 Audio loop started")
        
        async for transcript in self.stt.stream():
            if not self.running:
                break
            
            if transcript and transcript.strip():
                self.logger.info(f"📝 STT: {transcript}")
                await self.speech_queue.put(transcript)
    
    # =========================================
    # VISION LOOP
    # =========================================
    
    async def _vision_loop(self):
        """Pętla przetwarzania wideo."""
        self.logger.info("📷 Vision loop started")
        
        async for detections in self.detector.stream():
            if not self.running:
                break
            
            # Konwersja na DetectedObject
            objects = []
            for det in detections:
                obj = DetectedObject(
                    label=det["label"],
                    confidence=det["confidence"],
                    position=self._get_position(det["bbox"]),
                    bbox=det["bbox"]
                )
                objects.append(obj)
            
            # Zapisz do bufora
            self._detected_objects.append({
                "timestamp": time.time(),
                "objects": objects
            })
            
            # Debug log co 30 klatek
            if len(self._detected_objects) % 30 == 0 and objects:
                labels = [o.label for o in objects]
                self.logger.debug(f"👁️ Wykryto: {labels}")
    
    def _get_position(self, bbox: tuple) -> str:
        """Konwersja bbox na pozycję słowną."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Poziomo
        if cx < 0.33:
            h_pos = "lewo"
        elif cx > 0.66:
            h_pos = "prawo"
        else:
            h_pos = "środek"
        
        # Pionowo
        if cy < 0.33:
            v_pos = "góra"
        elif cy > 0.66:
            v_pos = "dół"
        else:
            v_pos = "środek"
        
        if h_pos == "środek" and v_pos == "środek":
            return "środek"
        elif v_pos == "środek":
            return h_pos
        elif h_pos == "środek":
            return v_pos
        else:
            return f"{v_pos}-{h_pos}"
    
    # =========================================
    # PROCESSING LOOP
    # =========================================
    
    async def _processing_loop(self):
        """Pętla przetwarzania zapytań."""
        self.logger.info("🧠 Processing loop started")
        
        while self.running:
            try:
                # Czekaj na tekst z STT (timeout 1s)
                transcript = await asyncio.wait_for(
                    self.speech_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            
            # Przetwórz zapytanie
            await self._process_query(transcript)
    
    async def _process_query(self, transcript: str):
        """Przetwarzanie pojedynczego zapytania."""
        self.logger.info(f"🔄 Przetwarzanie: {transcript}")
        
        # Sprawdź komendy systemowe
        if self._is_stop_command(transcript):
            self.logger.info("🛑 Komenda stop")
            await self.tts.speak("Do zobaczenia!")
            await self.stop()
            return
        
        # Pobierz ostatnie wykrycia
        current_objects = self._get_current_objects()
        
        # Zbuduj prompt
        prompt = self._build_prompt(transcript, current_objects)
        
        # Wyślij do LLM
        self.logger.debug(f"📤 Prompt: {prompt[:200]}...")
        response = await self.llm.generate(prompt)
        
        if response:
            self.logger.info(f"💬 LLM: {response}")
            
            # Zapisz kontekst
            self.context.last_speech = transcript
            self.context.last_response = response
            self.context.last_objects = current_objects
            self.context.timestamp = time.time()
            
            # TTS
            await self.tts.speak(response)
        else:
            self.logger.warning("Brak odpowiedzi z LLM")
            await self.tts.speak("Przepraszam, nie udało się przetworzyć zapytania.")
    
    def _is_stop_command(self, text: str) -> bool:
        """Sprawdzenie czy to komenda zatrzymania."""
        stop_words = ["stop", "koniec", "zakończ", "wyłącz", "do widzenia", "pa pa"]
        text_lower = text.lower().strip()
        return any(word in text_lower for word in stop_words)
    
    def _get_current_objects(self) -> List[DetectedObject]:
        """Pobranie ostatnich wykrytych obiektów."""
        if not self._detected_objects:
            return []
        
        # Ostatnie wykrycie
        latest = self._detected_objects[-1]
        return latest.get("objects", [])
    
    def _build_prompt(self, transcript: str, objects: List[DetectedObject]) -> str:
        """Budowanie promptu dla LLM."""
        # Lista obiektów
        if objects:
            obj_list = []
            for obj in objects:
                obj_list.append(f"[{obj.label}: {obj.position}, {obj.confidence*100:.0f}%]")
            objects_str = ", ".join(obj_list)
        else:
            objects_str = "[brak wykrytych obiektów w kadrze]"
        
        prompt = f"""Użytkownik pyta: "{transcript}"

Aktualne obiekty w polu widzenia kamery:
{objects_str}

Odpowiedz krótko i konkretnie po polsku."""
        
        return prompt
    
    # =========================================
    # PUBLIC API
    # =========================================
    
    async def query(self, text: str, include_vision: bool = True) -> str:
        """
        Programowe zapytanie (API).
        
        Args:
            text: Tekst zapytania
            include_vision: Czy uwzględnić dane z kamery
            
        Returns:
            Odpowiedź z LLM
        """
        objects = self._get_current_objects() if include_vision else []
        prompt = self._build_prompt(text, objects)
        return await self.llm.generate(prompt)
