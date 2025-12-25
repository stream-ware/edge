"""
MQTT Adapter - Komunikacja z urządzeniami IoT

Obsługuje:
- Publish/Subscribe
- Callbacks per topic
- Async integration
"""

import asyncio
import logging
from typing import Dict, Any, Callable, Optional, List
import json

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


class MQTTAdapter:
    """Adapter MQTT dla komunikacji IoT."""
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger("mqtt_adapter")
        
        self.broker = config.get("broker", "localhost")
        self.port = config.get("port", 1883)
        self.username = config.get("username")
        self.password = config.get("password")
        self.client_id = config.get("client_id", "streamware-orchestrator")
        
        self.client: Optional[mqtt.Client] = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._connected = False
        self._message_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self):
        """Połączenie z brokerem MQTT."""
        if mqtt is None:
            self.logger.warning("paho-mqtt not installed, MQTT disabled")
            return
        
        self.client = mqtt.Client(client_id=self.client_id)
        
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        
        # Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            
            # Czekaj na połączenie
            for _ in range(50):  # 5 sekund timeout
                if self._connected:
                    break
                await asyncio.sleep(0.1)
            
            if self._connected:
                self.logger.info(f"✅ MQTT connected to {self.broker}:{self.port}")
            else:
                self.logger.warning("MQTT connection timeout")
                
        except Exception as e:
            self.logger.error(f"MQTT connect error: {e}")
    
    async def disconnect(self):
        """Rozłączenie."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback przy połączeniu."""
        if rc == 0:
            self._connected = True
            self.logger.debug("MQTT connected")
            
            # Re-subscribe do wszystkich tematów
            for topic in self._callbacks.keys():
                client.subscribe(topic)
        else:
            self.logger.error(f"MQTT connect failed: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback przy rozłączeniu."""
        self._connected = False
        self.logger.warning(f"MQTT disconnected: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Callback przy wiadomości."""
        # Put to async queue
        asyncio.get_event_loop().call_soon_threadsafe(
            self._message_queue.put_nowait,
            (msg.topic, msg.payload.decode("utf-8", errors="replace"))
        )
    
    async def subscribe(self, topic: str, callback: Callable):
        """
        Subskrypcja tematu z callbackiem.
        
        Args:
            topic: Topic pattern (np. "commands/#")
            callback: Async callback(topic, payload)
        """
        if topic not in self._callbacks:
            self._callbacks[topic] = []
        
        self._callbacks[topic].append(callback)
        
        if self.client and self._connected:
            self.client.subscribe(topic)
            self.logger.debug(f"Subscribed to {topic}")
    
    async def publish(self, topic: str, payload: str):
        """
        Publikacja wiadomości.
        
        Args:
            topic: Topic
            payload: Payload (string lub JSON)
        """
        if not self.client or not self._connected:
            self.logger.warning("MQTT not connected, cannot publish")
            return
        
        self.client.publish(topic, payload)
        self.logger.debug(f"Published to {topic}: {payload[:100]}")
    
    async def loop(self):
        """Główna pętla przetwarzania wiadomości."""
        while True:
            try:
                topic, payload = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                # Znajdź pasujące callbacks
                for pattern, callbacks in self._callbacks.items():
                    if self._topic_matches(pattern, topic):
                        for callback in callbacks:
                            try:
                                await callback(topic, payload)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")
                                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
    
    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Sprawdza czy topic pasuje do wzorca (z # i +)."""
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")
        
        for i, pp in enumerate(pattern_parts):
            if pp == "#":
                return True
            if pp == "+":
                continue
            if i >= len(topic_parts) or pp != topic_parts[i]:
                return False
        
        return len(pattern_parts) == len(topic_parts)
    
    async def execute(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wykonanie akcji MQTT (dla spójności z innymi adapterami).
        """
        action = dsl.get("action", "")
        
        if action == "mqtt.publish":
            topic = dsl.get("topic", "events/default")
            payload = dsl.get("payload", "{}")
            await self.publish(topic, payload)
            return {"status": "ok"}
        
        return {"status": "error", "error": f"Unknown action: {action}"}
