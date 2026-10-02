"""Reescritura de la salida con las fusiones de identidad por número (SPEC-003 RF-4)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def resolve_chains(merges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """{origen: destino final} resolviendo cadenas X -> Y -> Z."""
    direct = dict(merges)
    out: dict[str, str] = {}
    for src in direct:
        dst, seen = direct[src], {src}
        while dst in direct and dst not in seen:
            seen.add(dst)
            dst = direct[dst]
        out[src] = dst
    return out


def apply_merges(jsonl: Path, merges: Iterable[tuple[str, str]]) -> int:
    """Reescribe `jsonl` reemplazando cada player_id fusionado por su destino. Devuelve cajas modificadas.

    Las identidades fusionadas nunca coexistieron en un frame (RF-3), así que no se generan duplicados.
    """
    mapping = resolve_chains(merges)
    if not mapping:
        return 0
    changed = 0
    lines = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for p in row["players"]:
            dst = mapping.get(p["player_id"])
            if dst is not None:
                p["player_id"] = dst
                changed += 1
        lines.append(json.dumps(row, ensure_ascii=False))
    jsonl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return changed
