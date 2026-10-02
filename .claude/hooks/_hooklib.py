"""Utilidades compartidas por los hooks del harness (solo stdlib: corren fuera del venv).

Protocolo Claude Code: el hook recibe JSON por stdin. Para PreToolUse responde por stdout
``{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny"|"ask", ...}}``
y sale con código 0. Sin salida = sin objeción (sigue el flujo normal de permisos).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

DecisionKind = Literal["deny", "ask"]


@dataclass(frozen=True)
class Decision:
    decision: DecisionKind
    reason: str


def strongest(decisions: list[Decision | None]) -> Decision | None:
    found = [d for d in decisions if d is not None]
    for kind in ("deny", "ask"):
        for d in found:
            if d.decision == kind:
                return d
    return None


def read_payload() -> dict[str, Any] | None:
    try:
        data = json.loads(sys.stdin.read() or "null")
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())


def log_event(event: dict[str, Any]) -> None:
    """Telemetría JSONL. Nunca registra contenido de comandos, prompts ni archivos."""
    log_dir = Path(os.environ.get("HARNESS_LOG_DIR") or project_dir() / ".harness" / "logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **event}
        with (log_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # la telemetría nunca debe romper el flujo


def emit_pre_tool_use(decision: Decision | None, hook: str, payload: dict[str, Any] | None) -> None:
    payload = payload or {}
    if decision is not None:
        log_event(
            {
                "event": "guard_decision",
                "hook": hook,
                "tool": payload.get("tool_name"),
                "session": payload.get("session_id"),
                "decision": decision.decision,
                "reason": decision.reason,
            }
        )
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision.decision,
                "permissionDecisionReason": f"[harness:{hook}] {decision.reason}",
            }
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.exit(0)
