"""Resume la telemetría del harness (.harness/logs/events.jsonl) para el informe de sprint.

Uso: python scripts/harness_metrics.py [events.jsonl] [--since AAAA-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    tool_uses = [r for r in records if r.get("event") == "PostToolUse"]
    guards = [r for r in records if r.get("event") == "guard_decision"]
    errors = [r for r in tool_uses if r.get("error") is True]
    return {
        "sessions": len({r.get("session") for r in records if r.get("session")}),
        "tool_uses": len(tool_uses),
        "tool_error_rate": round(len(errors) / len(tool_uses), 3) if tool_uses else None,
        "tools": dict(Counter(r.get("tool") for r in tool_uses).most_common()),
        "guard_decisions": dict(
            Counter(f"{r.get('decision')}: {r.get('reason')}" for r in guards).most_common()
        ),
        "lint_blocks": sum(1 for r in records if r.get("event") == "lint_block"),
    }


def load(path: Path, since: str | None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is None or str(rec.get("ts", "")) >= since:
            out.append(rec)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, nargs="?", default=Path(".harness/logs/events.jsonl"))
    ap.add_argument("--since")
    args = ap.parse_args(argv)
    records = load(args.log, args.since)
    if not records:
        print(f"Sin telemetría en {args.log} (desconocido, no cero).")
        return 0
    print(json.dumps(summarize(records), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
