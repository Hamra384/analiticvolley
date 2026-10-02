"""Votación de dorsal a nivel identidad (SPEC-001 RF-7). Ante duda, None."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from volley_cv.identity.settings import IdentityConfig
from volley_cv.identity.types import JerseyRead


@dataclass
class JerseyVotes:
    counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    conf_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, read: JerseyRead) -> None:
        self.counts[read.number] += 1
        self.conf_sum[read.number] += read.confidence

    def remove(self, read: JerseyRead) -> None:
        """Revierte una lectura atribuida a esta identidad por error (re-vinculación de tracklets)."""
        if self.counts.get(read.number, 0) <= 0:
            return
        self.counts[read.number] -= 1
        self.conf_sum[read.number] -= read.confidence
        if self.counts[read.number] == 0:
            del self.counts[read.number]
            del self.conf_sum[read.number]

    def candidate(self, cfg: IdentityConfig) -> tuple[int, float, float] | None:
        """(número, confianza media, evidencia) si cumple los umbrales; si no, None."""
        if not self.counts:
            return None
        total = sum(self.conf_sum.values())
        number = max(self.counts, key=lambda n: (self.conf_sum[n], self.counts[n], -n))
        count, conf_sum = self.counts[number], self.conf_sum[number]
        mean_conf = conf_sum / count
        if count < cfg.jersey_min_reads or mean_conf < cfg.jersey_min_conf:
            return None
        if total <= 0 or conf_sum / total < cfg.jersey_min_share:
            return None
        return number, mean_conf, conf_sum


def resolve_team_conflicts(
    candidates: dict[str, tuple[int, float, float] | None],
) -> dict[str, tuple[int, float] | None]:
    """Unicidad por equipo: si varias identidades reclaman el mismo número, gana la de mayor evidencia."""
    best: dict[int, tuple[float, str]] = {}
    for pid, cand in candidates.items():
        if cand is None:
            continue
        number, _, evidence = cand
        current = best.get(number)
        if current is None or (evidence, pid) > current:  # desempate determinista
            best[number] = (evidence, pid)
    out: dict[str, tuple[int, float] | None] = {}
    for pid, cand in candidates.items():
        if cand is not None and best[cand[0]][1] == pid:
            out[pid] = (cand[0], cand[1])
        else:
            out[pid] = None
    return out
