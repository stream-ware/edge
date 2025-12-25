#!/usr/bin/env python3
"""
Firmware Simulator - Symulacja urządzeń IoT/Edge

Generuje dane sensorów i wysyła przez MQTT.
"""

import os
import time
import json
import random
import logging
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Installing paho-mqtt...")
    import subprocess
    subprocess.check_call(["pip", "install", "paho-mqtt"])
    import paho.mqtt.client as mqtt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("firmware_sim")


class SensorSimulator:
    """Symulator sensorów IoT."""
    
    def __init__(self):
        self.broker = os.getenv("MQTT_BROKER", "localhost")
        self.port = int(os.getenv("MQTT_PORT", "1883"))
        self.interval = int(os.getenv("SENSOR_INTERVAL", "5"))
        
        self.client = mqtt.Client(client_id="firmware-simulator")
        self.connected = False
        
        # Symulowane lokalizacje
        self.locations = ["salon", "kuchnia", "sypialnia", "lazienka", "garaz"]
        
        # Stan urządzeń
        self.devices = {
            loc: {
                "temperature": 20 + random.uniform(0, 5),
                "humidity": 40 + random.uniform(0, 20),
                "light_level": random.randint(0, 1000),
                "motion": False
            }
            for loc in self.locations
        }
    
    def connect(self):
        """Połączenie z brokerem MQTT."""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                self.connected = True
                logger.info(f"Connected to MQTT broker {self.broker}:{self.port}")
                # Subscribe na komendy
                client.subscribe("edge/commands/#")
            else:
                logger.error(f"Connection failed: {rc}")
        
        def on_message(client, userdata, msg):
            self.handle_command(msg.topic, msg.payload.decode())
        
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            
            # Wait for connection
            for _ in range(50):
                if self.connected:
                    break
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Cannot connect: {e}")
    
    def handle_command(self, topic: str, payload: str):
        """Obsługa komend z orchestratora."""
        logger.info(f"Received command: {topic} -> {payload}")
        
        try:
            data = json.loads(payload)
            # Obsłuż komendy sterujące
            # np. edge/commands/salon/light -> {"state": "on"}
        except json.JSONDecodeError:
            pass
    
    def simulate_sensors(self):
        """Symulacja odczytów sensorów."""
        for location, sensors in self.devices.items():
            # Symulacja zmian wartości
            sensors["temperature"] += random.uniform(-0.5, 0.5)
            sensors["temperature"] = max(15, min(30, sensors["temperature"]))
            
            sensors["humidity"] += random.uniform(-2, 2)
            sensors["humidity"] = max(20, min(80, sensors["humidity"]))
            
            sensors["light_level"] = random.randint(0, 1000)
            sensors["motion"] = random.random() < 0.1  # 10% szansa na ruch
            
            # Publikuj dane
            for metric, value in sensors.items():
                topic = f"edge/sensors/{location}/{metric}"
                payload = json.dumps({
                    "value": round(value, 2) if isinstance(value, float) else value,
                    "timestamp": datetime.now().isoformat(),
                    "device_id": f"{location}-sensor"
                })
                
                self.client.publish(topic, payload)
        
        logger.debug(f"Published sensor data for {len(self.locations)} locations")
    
    def run(self):
        """Główna pętla."""
        logger.info(f"Starting sensor simulation (interval: {self.interval}s)")
        
        while True:
            if self.connected:
                self.simulate_sensors()
            time.sleep(self.interval)


def main():
    simulator = SensorSimulator()
    simulator.connect()
    
    try:
        simulator.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        simulator.client.loop_stop()
        simulator.client.disconnect()


if __name__ == "__main__":
    main()
