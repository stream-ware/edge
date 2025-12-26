#!/usr/bin/env python3
"""
Shell Adapter - bezpieczne wykonywanie komend diagnostycznych.

Workflow: informacja → analiza → rozwiązanie
- Whitelist dozwolonych komend
- Timeout i limity
- Formatowanie wyników dla LLM
"""

import asyncio
import logging
import shlex
import os
import time
from typing import Optional, Dict, Any, List


class ShellAdapter:
    """Adapter do bezpiecznego wykonywania komend shell."""
    
    ALLOWED_COMMANDS = {
        # Diagnostyka systemu
        "systemctl": ["status", "is-active", "list-units"],
        "journalctl": ["-u", "-n", "--no-pager"],
        "ping": ["-c"],
        "curl": ["-s", "-I", "--connect-timeout"],
        "nc": ["-zv", "-w"],
        "netstat": ["-tlnp", "-ulnp"],
        "ss": ["-tlnp", "-ulnp"],
        "lsof": ["-i"],
        "ps": ["aux", "-ef"],
        "top": ["-bn1"],
        "free": ["-h"],
        "df": ["-h"],
        "uptime": [],
        "hostname": [],
        "uname": ["-a"],
        "cat": ["/etc/os-release", "/proc/cpuinfo", "/proc/meminfo"],
        "ls": ["-la", "-l"],
        "which": [],
        "whereis": [],
        "file": [],
        "head": ["-n"],
        "tail": ["-n", "-f"],
        "grep": [],
        "awk": [],
        "wc": ["-l"],
        "date": [],
        "env": [],
        "printenv": [],
        # Docker diagnostyka
        "docker": ["ps", "logs", "inspect", "stats", "info", "version", "network"],
        "docker-compose": ["ps", "logs", "config"],
        # MQTT diagnostyka
        "mosquitto_sub": ["-t", "-C", "-W"],
        "mosquitto_pub": ["-t", "-m"],
        # Sieć
        "ip": ["addr", "route", "link"],
        "ifconfig": [],
        "nslookup": [],
        "dig": [],
        "traceroute": [],
        # Sensory / hardware
        "lsusb": [],
        "lspci": [],
        "dmesg": ["-T", "--level"],
        "i2cdetect": ["-y"],
        "vcgencmd": ["measure_temp", "get_throttled"],
        # Python / pip
        "pip": ["list", "show", "freeze"],
        "python": ["--version", "-c"],
    }
    
    DANGEROUS_PATTERNS = [
        "rm ", "rm\t", "rmdir",
        "> /", ">> /",
        "sudo ", "su ",
        "chmod ", "chown ",
        "mkfs", "dd ",
        ":(){ :", "fork",
        "; rm", "&& rm",
        "| rm", "`rm",
        "$(rm",
        "eval ", "exec ",
        "shutdown", "reboot", "halt",
        "kill ", "killall ",
        "passwd", "useradd", "userdel",
    ]
    
    def __init__(self, config: dict = None):
        self.logger = logging.getLogger("shell_adapter")
        self.config = config or {}
        self.timeout = self.config.get("timeout", 10)
        self.max_output = self.config.get("max_output", 4000)
        self.cwd = self.config.get("cwd", os.getcwd())
    
    async def initialize(self):
        self.logger.info("✅ Shell Adapter initialized")
    
    async def cleanup(self):
        pass
    
    def _is_safe_command(self, cmd: str) -> tuple[bool, str]:
        """Sprawdza czy komenda jest bezpieczna."""
        cmd_lower = cmd.lower()
        
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in cmd_lower:
                return False, f"Niebezpieczny wzorzec: {pattern}"
        
        parts = shlex.split(cmd)
        if not parts:
            return False, "Pusta komenda"
        
        base_cmd = os.path.basename(parts[0])
        
        if base_cmd not in self.ALLOWED_COMMANDS:
            return False, f"Komenda '{base_cmd}' nie jest na liście dozwolonych"
        
        return True, "OK"
    
    async def execute(self, command: str, timeout: int = None) -> Dict[str, Any]:
        """Wykonuje komendę shell z walidacją bezpieczeństwa."""
        timeout = timeout or self.timeout

        start = time.perf_counter()
        
        is_safe, reason = self._is_safe_command(command)
        if not is_safe:
            self.logger.warning(f"Odrzucono komendę: {command} - {reason}")
            return {
                "status": "rejected",
                "command": command,
                "error": reason,
                "suggestion": "Użyj dozwolonej komendy diagnostycznej"
            }
        
        self.logger.info(f"shell.run start: {command}")
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                duration_ms = int((time.perf_counter() - start) * 1000)
                self.logger.warning(f"shell.run timeout after {duration_ms}ms: {command}")
                return {
                    "status": "timeout",
                    "command": command,
                    "error": f"Przekroczono limit czasu ({timeout}s)",
                    "duration_ms": duration_ms,
                    "suggestion": "Komenda trwa zbyt długo, spróbuj z mniejszym zakresem"
                }
            
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            
            if len(stdout_text) > self.max_output:
                stdout_text = stdout_text[:self.max_output] + "\n... (obcięto)"
            if len(stderr_text) > self.max_output:
                stderr_text = stderr_text[:self.max_output] + "\n... (obcięto)"

            duration_ms = int((time.perf_counter() - start) * 1000)
            stdout_lines = len(stdout_text.splitlines()) if stdout_text else 0
            stderr_lines = len(stderr_text.splitlines()) if stderr_text else 0

            self.logger.info(
                f"shell.run done: rc={proc.returncode} duration_ms={duration_ms} stdout_lines={stdout_lines} stderr_lines={stderr_lines} cmd={command}"
            )
            
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "command": command,
                "returncode": proc.returncode,
                "duration_ms": duration_ms,
                "stdout_lines": stdout_lines,
                "stderr_lines": stderr_lines,
                "stdout": stdout_text.strip(),
                "stderr": stderr_text.strip(),
            }
            
        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self.logger.error(f"Shell error: {e}")
            return {
                "status": "error",
                "command": command,
                "duration_ms": duration_ms,
                "error": str(e)
            }
    
    async def diagnose(self, topic: str) -> Dict[str, Any]:
        """Uruchamia zestaw komend diagnostycznych dla tematu."""
        diagnostics = {
            "mqtt": [
                "systemctl is-active mosquitto || echo 'not-systemd'",
                "docker ps --filter name=mqtt --format '{{.Names}}: {{.Status}}'",
                "ss -tlnp | grep :1883 || echo 'port 1883 not listening'",
                "ping -c 1 localhost > /dev/null && echo 'localhost OK' || echo 'localhost FAIL'",
            ],
            "docker": [
                "docker info --format '{{.ServerVersion}}' 2>/dev/null || echo 'docker not running'",
                "docker ps --format '{{.Names}}: {{.Status}}' | head -10",
                "systemctl is-active docker || echo 'not-systemd'",
            ],
            "network": [
                "ip addr show | grep 'inet ' | head -5",
                "ping -c 1 8.8.8.8 > /dev/null && echo 'internet OK' || echo 'no internet'",
                "cat /etc/resolv.conf | grep nameserver | head -3",
            ],
            "sensors": [
                "ls /dev/i2c* 2>/dev/null || echo 'no i2c devices'",
                "ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo 'no serial devices'",
                "lsusb 2>/dev/null | head -10",
                "vcgencmd measure_temp 2>/dev/null || cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 'no temp sensor'",
            ],
            "audio": [
                "aplay -l 2>/dev/null | head -10 || echo 'no audio devices'",
                "arecord -l 2>/dev/null | head -10 || echo 'no recording devices'",
                "pulseaudio --check 2>/dev/null && echo 'pulseaudio running' || echo 'pulseaudio not running'",
            ],
            "camera": [
                "ls /dev/video* 2>/dev/null || echo 'no video devices'",
                "v4l2-ctl --list-devices 2>/dev/null || echo 'v4l2-ctl not available'",
            ],
            "system": [
                "uptime",
                "free -h | head -2",
                "df -h / | tail -1",
                "uname -a",
            ],
        }
        
        commands = diagnostics.get(topic, diagnostics["system"])
        results = []
        
        for cmd in commands:
            result = await self.execute(cmd, timeout=5)
            results.append(result)
        
        return {
            "topic": topic,
            "diagnostics": results
        }
    
    def get_diagnostic_topics(self) -> List[str]:
        """Zwraca dostępne tematy diagnostyczne."""
        return ["mqtt", "docker", "network", "sensors", "audio", "camera", "system"]
