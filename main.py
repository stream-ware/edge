#!/usr/bin/env python3
"""
Streamware Jetson - Lokalny Asystent Wizyjno-Głosowy
Main entry point

Author: Softreck / prototypowanie.pl
License: MIT
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path

import yaml

from src.orchestrator import Orchestrator


def setup_logging(config: dict) -> logging.Logger:
    """Konfiguracja logowania."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    
    # Utworzenie katalogu logs
    log_file = log_config.get("file", "logs/streamware.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Formatowanie
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler konsoli
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Handler pliku
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger("streamware")


def load_config(config_path: str = "config.yaml") -> dict:
    """Wczytanie konfiguracji z pliku YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def print_banner():
    """Wyświetlenie bannera startowego."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤖 STREAMWARE JETSON                                    ║
    ║   Lokalny Asystent Wizyjno-Głosowy                        ║
    ║                                                           ║
    ║   Softreck / prototypowanie.pl                            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


async def main():
    """Główna funkcja asynchroniczna."""
    print_banner()
    
    # Wczytanie konfiguracji
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Uruchamianie Streamware Jetson...")
    logger.info("=" * 60)
    
    # Utworzenie orchestratora
    orchestrator = Orchestrator(config)
    
    # Obsługa sygnałów (Ctrl+C)
    loop = asyncio.get_running_loop()
    
    def signal_handler():
        logger.info("Otrzymano sygnał zatrzymania...")
        asyncio.create_task(orchestrator.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Inicjalizacja komponentów
        logger.info("Inicjalizacja komponentów...")
        await orchestrator.initialize()
        
        logger.info("✅ System gotowy!")
        logger.info("Powiedz coś lub naciśnij Ctrl+C aby zakończyć")
        print("\n" + "=" * 60)
        print("🎤 Nasłuchuję... (Ctrl+C = stop)")
        print("=" * 60 + "\n")
        
        # Główna pętla
        await orchestrator.run()
        
    except Exception as e:
        logger.exception(f"Błąd krytyczny: {e}")
        sys.exit(1)
    finally:
        await orchestrator.cleanup()
        logger.info("Streamware zakończony.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Do zobaczenia!")
