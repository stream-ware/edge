"""Env Adapter - zarządzanie zmiennymi w pliku .env"""

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EnvItem:
    key: str
    value: str


class EnvAdapter:
    """Adapter do zarządzania zmiennymi środowiskowymi poprzez plik .env."""

    _key_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _line_re = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.env_path = Path(config.get("path", ".env"))
        self.editor = config.get("editor")

    async def initialize(self):
        return

    async def execute(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        action = dsl.get("action", "")

        if action == "env.get":
            key = dsl.get("key")
            return self._get(key)

        if action == "env.set":
            key = dsl.get("key")
            value = dsl.get("value")
            return self._set(key, value)

        if action == "env.unset":
            key = dsl.get("key")
            return self._unset(key)

        if action == "env.list":
            return self._list()

        if action == "env.editor":
            return self._open_editor()

        if action == "env.reload":
            return self._reload_into_process()

        return {"status": "error", "error": f"Unknown action: {action}"}

    def _validate_key(self, key: Optional[str]) -> Tuple[bool, Optional[str]]:
        if not key:
            return False, "No key specified"
        if not self._key_re.match(key):
            return False, "Invalid key format"
        return True, None

    def _read_lines(self) -> List[str]:
        if not self.env_path.exists():
            return []
        return self.env_path.read_text(encoding="utf-8").splitlines(keepends=True)

    def _write_lines(self, lines: List[str]) -> None:
        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        self.env_path.write_text("".join(lines), encoding="utf-8")

    def _parse(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for raw in self._read_lines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = self._line_re.match(line)
            if not m:
                continue
            key, value = m.group(1), m.group(2)
            result[key] = value
        return result

    def _get(self, key: Optional[str]) -> Dict[str, Any]:
        ok, err = self._validate_key(key)
        if not ok:
            return {"status": "error", "error": err}

        items = self._parse()
        if key not in items:
            return {"status": "error", "error": f"Key '{key}' not found"}

        return {"status": "ok", "key": key, "value": items[key]}

    def _set(self, key: Optional[str], value: Any) -> Dict[str, Any]:
        ok, err = self._validate_key(key)
        if not ok:
            return {"status": "error", "error": err}

        if value is None:
            return {"status": "error", "error": "No value specified"}

        value_str = str(value)

        lines = self._read_lines()
        updated = False

        for i, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue

            m = self._line_re.match(stripped)
            if not m:
                continue

            if m.group(1) == key:
                newline = "\n" if raw.endswith("\n") else ""
                lines[i] = f"{key}={value_str}{newline}"
                updated = True
                break

        if not updated:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] = lines[-1] + "\n"
            lines.append(f"{key}={value_str}\n")

        self._write_lines(lines)
        os.environ[key] = value_str

        return {"status": "ok", "key": key, "value": value_str, "persisted": True}

    def _unset(self, key: Optional[str]) -> Dict[str, Any]:
        ok, err = self._validate_key(key)
        if not ok:
            return {"status": "error", "error": err}

        lines = self._read_lines()
        new_lines: List[str] = []
        removed = False

        for raw in lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(raw)
                continue

            m = self._line_re.match(stripped)
            if m and m.group(1) == key:
                removed = True
                continue

            new_lines.append(raw)

        if not removed:
            return {"status": "error", "error": f"Key '{key}' not found"}

        self._write_lines(new_lines)
        os.environ.pop(key, None)
        return {"status": "ok", "key": key, "removed": True}

    def _list(self) -> Dict[str, Any]:
        items = self._parse()
        keys = sorted(items.keys())
        return {"status": "ok", "count": len(keys), "keys": keys}

    def _open_editor(self) -> Dict[str, Any]:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return {
                "status": "error",
                "error": "No TTY available for interactive editor",
            }

        self.env_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.env_path.exists():
            self.env_path.write_text("", encoding="utf-8")

        editor = self.editor or os.environ.get("EDITOR")

        if editor:
            try:
                args = shlex.split(editor)
                if args and shutil.which(args[0]):
                    subprocess.run([*args, str(self.env_path)], check=False)
                    return {"status": "ok", "path": str(self.env_path), "editor": editor}
            except Exception:
                pass

        return self._interactive_editor()

    def _interactive_editor(self) -> Dict[str, Any]:
        print(f"Editing {self.env_path} (builtin editor). Type 'help' for commands.")

        while True:
            try:
                cmd = input("env> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                break

            if not cmd:
                continue

            low = cmd.lower()
            if low in {"exit", "quit", "q"}:
                break

            if low in {"help", "h", "?"}:
                print("Commands: list | get KEY | set KEY VALUE | unset KEY | reload | exit")
                continue

            if low == "list":
                res = self._list()
                if res.get("status") == "ok":
                    for k in res.get("keys", []):
                        print(k)
                else:
                    print(res.get("error", "error"))
                continue

            if low.startswith("get "):
                parts = cmd.split(None, 1)
                key = parts[1].strip() if len(parts) > 1 else None
                res = self._get(key)
                if res.get("status") == "ok":
                    print(f"{res.get('key')}={res.get('value')}")
                else:
                    print(res.get("error", "error"))
                continue

            if low.startswith("unset "):
                parts = cmd.split(None, 1)
                key = parts[1].strip() if len(parts) > 1 else None
                res = self._unset(key)
                if res.get("status") == "ok":
                    print(f"Removed {res.get('key')}")
                else:
                    print(res.get("error", "error"))
                continue

            if low.startswith("set "):
                parts = cmd.split(None, 2)
                if len(parts) < 3:
                    print("Usage: set KEY VALUE")
                    continue
                key = parts[1].strip()
                value = parts[2]
                res = self._set(key, value)
                if res.get("status") == "ok":
                    print(f"Set {res.get('key')}={res.get('value')}")
                else:
                    print(res.get("error", "error"))
                continue

            if low == "reload":
                res = self._reload_into_process()
                if res.get("status") == "ok":
                    print(f"Reloaded {res.get('count', 0)} vars")
                else:
                    print(res.get("error", "error"))
                continue

            print("Unknown command. Type 'help'.")

        return {"status": "ok", "path": str(self.env_path), "editor": "builtin"}

    def _reload_into_process(self) -> Dict[str, Any]:
        # Minimal reload: parse file and inject into os.environ
        items = self._parse()
        for k, v in items.items():
            os.environ[k] = v
        return {"status": "ok", "count": len(items)}
