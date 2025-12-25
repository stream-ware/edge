"""
Docker Adapter - Zarządzanie kontenerami Docker

Obsługuje:
- restart, stop, start
- logs
- status, inspect
- list
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

try:
    import docker
    from docker.errors import NotFound, APIError
except ImportError:
    docker = None


class DockerAdapter:
    """Adapter do Docker API."""
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("docker_adapter")
        self.config = config or {}
        self.client = None
    
    async def initialize(self):
        """Inicjalizacja klienta Docker."""
        if docker is None:
            self.logger.warning("docker package not installed")
            return
        
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            self.logger.info("✅ Docker connected")
        except Exception as e:
            self.logger.error(f"Cannot connect to Docker: {e}")
            self.client = None
    
    async def cleanup(self):
        """Zamknięcie klienta."""
        if self.client:
            self.client.close()
    
    async def execute(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wykonanie akcji Docker.
        
        Args:
            dsl: Komenda DSL z action i parametrami
            
        Returns:
            Wynik operacji
        """
        if not self.client:
            return {"status": "error", "error": "Docker not connected"}
        
        action = dsl.get("action", "")
        target = dsl.get("target")
        
        # Router akcji
        handlers = {
            "docker.restart": self._restart,
            "docker.stop": self._stop,
            "docker.start": self._start,
            "docker.logs": self._logs,
            "docker.status": self._status,
            "docker.inspect": self._inspect,
            "docker.list": self._list,
        }
        
        handler = handlers.get(action)
        
        if not handler:
            return {"status": "error", "error": f"Unknown action: {action}"}
        
        # Execute in thread pool (docker-py is sync)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handler, dsl)
    
    def _restart(self, dsl: dict) -> dict:
        """Restart kontenera."""
        target = dsl.get("target")
        if not target:
            return {"status": "error", "error": "No target specified"}
        
        try:
            container = self.client.containers.get(target)
            container.restart(timeout=10)
            return {"status": "ok", "target": target}
        except NotFound:
            return {"status": "error", "error": f"Container '{target}' not found"}
        except APIError as e:
            return {"status": "error", "error": str(e)}
    
    def _stop(self, dsl: dict) -> dict:
        """Stop kontenera."""
        target = dsl.get("target")
        if not target:
            return {"status": "error", "error": "No target specified"}
        
        try:
            container = self.client.containers.get(target)
            container.stop(timeout=10)
            return {"status": "ok", "target": target}
        except NotFound:
            return {"status": "error", "error": f"Container '{target}' not found"}
        except APIError as e:
            return {"status": "error", "error": str(e)}
    
    def _start(self, dsl: dict) -> dict:
        """Start kontenera."""
        target = dsl.get("target")
        if not target:
            return {"status": "error", "error": "No target specified"}
        
        try:
            container = self.client.containers.get(target)
            container.start()
            return {"status": "ok", "target": target}
        except NotFound:
            return {"status": "error", "error": f"Container '{target}' not found"}
        except APIError as e:
            return {"status": "error", "error": str(e)}
    
    def _logs(self, dsl: dict) -> dict:
        """Pobierz logi kontenera."""
        target = dsl.get("target")
        tail = dsl.get("tail", 10)
        
        if not target:
            return {"status": "error", "error": "No target specified"}
        
        try:
            container = self.client.containers.get(target)
            logs = container.logs(tail=tail, timestamps=False).decode("utf-8", errors="replace")
            return {"status": "ok", "target": target, "logs": logs}
        except NotFound:
            return {"status": "error", "error": f"Container '{target}' not found"}
        except APIError as e:
            return {"status": "error", "error": str(e)}
    
    def _status(self, dsl: dict) -> dict:
        """Status wszystkich kontenerów."""
        try:
            containers = self.client.containers.list(all=True)
            
            result = []
            for c in containers:
                result.append({
                    "name": c.name,
                    "status": c.status,
                    "image": c.image.tags[0] if c.image.tags else "unknown"
                })
            
            return {"status": "ok", "containers": result}
        except APIError as e:
            return {"status": "error", "error": str(e)}
    
    def _inspect(self, dsl: dict) -> dict:
        """Szczegóły kontenera."""
        target = dsl.get("target")
        if not target:
            return {"status": "error", "error": "No target specified"}
        
        try:
            container = self.client.containers.get(target)
            
            return {
                "status": "ok",
                "target": target,
                "container_status": container.status,
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "ports": container.ports,
                "created": str(container.attrs.get("Created", ""))
            }
        except NotFound:
            return {"status": "error", "error": f"Container '{target}' not found"}
        except APIError as e:
            return {"status": "error", "error": str(e)}
    
    def _list(self, dsl: dict) -> dict:
        """Lista kontenerów."""
        try:
            containers = self.client.containers.list(all=True)
            
            result = [{"name": c.name, "status": c.status} for c in containers]
            
            return {"status": "ok", "containers": result}
        except APIError as e:
            return {"status": "error", "error": str(e)}
