"""Telemetría (PostToolUse, Stop, SessionStart): registra metadatos, nunca contenido.

Se guarda en .harness/logs/events.jsonl (gitignored). Sirve para las métricas del harness:
uso de herramientas por sesión, errores de herramientas y frecuencia de bloqueos
(ver docs/harness/METRICS.md).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hooklib import log_event, read_payload


def _is_error(response: Any) -> bool | None:
    if not isinstance(response, dict):
        return None
    if response.get("is_error") or response.get("interrupted"):
        return True
    if isinstance(response.get("success"), bool):
        return not response["success"]
    return False


def main() -> None:
    payload = read_payload() or {}
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    file_path = tool_input.get("file_path") if isinstance(tool_input, dict) else None
    log_event(
        {
            "event": payload.get("hook_event_name", "unknown"),
            "session": payload.get("session_id"),
            "tool": payload.get("tool_name"),
            "error": _is_error(payload.get("tool_response")),
            "file_ext": Path(file_path).suffix if isinstance(file_path, str) else None,
        }
    )


if __name__ == "__main__":
    main()
