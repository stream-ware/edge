"""
Firmware Adapter - Komunikacja z urządzeniami IoT/Edge

Obsługuje:
- Odczyt sensorów
- Sterowanie urządzeniami
- Integracja przez MQTT
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import json
import random  # Dla symulacji


class FirmwareAdapter:
    """Adapter dla urządzeń IoT/Edge."""
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("firmware_adapter")
        self.config = config or {}
        
        # Symulowane urządzenia (w rzeczywistości z MQTT)
        self._devices: Dict[str, Dict[str, Any]] = {
            "salon": {
                "temperature": 22.5,
                "humidity": 45,
                "light": "off"
            },
            "kuchnia": {
                "temperature": 23.0,
                "humidity": 50,
                "light": "off"
            },
            "sypialnia": {
                "temperature": 21.0,
                "humidity": 40,
                "light": "off"
            },
            "default": {
                "temperature": 22.0,
                "humidity": 45,
                "light": "off"
            }
        }
        
        # MQTT client (opcjonalne)
        self._mqtt = None
    
    def set_mqtt_client(self, mqtt_adapter):
        """Ustawienie klienta MQTT do komunikacji z rzeczywistymi urządzeniami."""
        self._mqtt = mqtt_adapter
    
    async def execute(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wykonanie akcji na urządzeniu.
        """
        action = dsl.get("action", "")
        
        handlers = {
            "sensor.read": self._read_sensor,
            "device.set": self._set_device,
            "device.get": self._get_device,
        }
        
        handler = handlers.get(action)
        
        if not handler:
            return {"status": "error", "error": f"Unknown action: {action}"}
        
        return await handler(dsl)
    
    async def _read_sensor(self, dsl: dict) -> dict:
        """Odczyt wartości sensora."""
        metric = dsl.get("metric", "temperature")
        location = dsl.get("location", dsl.get("device", "default"))
        
        # Znajdź urządzenie
        device = self._devices.get(location, self._devices["default"])
        
        # Pobierz wartość (z symulacją szumu)
        value = device.get(metric)
        
        if value is None:
            return {
                "status": "error",
                "error": f"Unknown metric '{metric}' for {location}"
            }
        
        # Dodaj szum dla realizmu (symulacja)
        if isinstance(value, (int, float)):
            value = round(value + random.uniform(-0.5, 0.5), 1)
        
        # Jednostki
        units = {
            "temperature": "°C",
            "humidity": "%",
            "pressure": "hPa",
            "light_level": "lux"
        }
        
        return {
            "status": "ok",
            "metric": metric,
            "value": value,
            "unit": units.get(metric, ""),
            "location": location
        }
    
    async def _set_device(self, dsl: dict) -> dict:
        """Ustawienie wartości urządzenia."""
        device_type = dsl.get("device", "")
        location = dsl.get("location", "default")
        state = dsl.get("state", dsl.get("value"))
        
        if not device_type:
            return {"status": "error", "error": "No device specified"}
        
        # Znajdź urządzenie
        device = self._devices.get(location)
        
        if device is None:
            return {"status": "error", "error": f"Unknown location: {location}"}
        
        # Mapowanie device_type na klucz
        key_map = {
            "light": "light",
            "światło": "light",
            "temp": "temperature",
            "temperatura": "temperature"
        }
        
        key = key_map.get(device_type, device_type)
        
        if key not in device:
            return {"status": "error", "error": f"Device '{device_type}' not found in {location}"}
        
        # Ustaw wartość
        device[key] = state
        
        # Publikuj na MQTT (jeśli dostępne)
        if self._mqtt:
            await self._mqtt.publish(
                f"edge/devices/{location}/{key}",
                json.dumps({"value": state})
            )
        
        self.logger.info(f"Set {location}/{key} = {state}")
        
        return {
            "status": "ok",
            "device": device_type,
            "location": location,
            "state": state
        }
    
    async def _get_device(self, dsl: dict) -> dict:
        """Pobranie stanu urządzenia."""
        location = dsl.get("location", "default")
        
        device = self._devices.get(location)
        
        if device is None:
            return {"status": "error", "error": f"Unknown location: {location}"}
        
        return {
            "status": "ok",
            "location": location,
            "state": device.copy()
        }
    
    def update_from_mqtt(self, topic: str, payload: str):
        """
        Aktualizacja stanu z MQTT (dla rzeczywistych urządzeń).
        
        Topic format: edge/sensors/{location}/{metric}
        """
        parts = topic.split("/")
        
        if len(parts) >= 4 and parts[0] == "edge" and parts[1] == "sensors":
            location = parts[2]
            metric = parts[3]
            
            try:
                data = json.loads(payload)
                value = data.get("value")
                
                if location not in self._devices:
                    self._devices[location] = {}
                
                self._devices[location][metric] = value
                self.logger.debug(f"Updated {location}/{metric} = {value}")
                
            except json.JSONDecodeError:
                pass
