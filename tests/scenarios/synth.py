"""Generador de escenarios sintéticos con ground truth para el IdentityManager (SPEC-001).

Simula lo que el tracker real hace mal (SPIKE-001): IDs nuevos tras huecos, intercambio de IDs en cruces y
fallos del detector. La apariencia de cada jugador = componente de equipo + componente individual + ruido,
de modo que compañeros de equipo se parecen más entre sí que a rivales (como con embeddings reales).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from volley_cv.identity import IdentityManager, JerseyRead, Observation, Team
from volley_cv.output.schema import FrameOutput

Path = Callable[[int], tuple[float, float] | None]
Box = tuple[float, float, float, float]


def linear(start: tuple[float, float], end: tuple[float, float], f0: int, f1: int) -> Path:
    def at(f: int) -> tuple[float, float] | None:
        if f < f0 or f > f1:
            return None
        t = (f - f0) / max(f1 - f0, 1)
        return start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])

    return at


def still(pos: tuple[float, float], f0: int = 0, f1: int = 10**9) -> Path:
    return linear(pos, pos, f0, f1)


@dataclass
class SimPlayer:
    gt: str
    team: Team
    path: Path
    height: float = 150.0
    jersey: Callable[[int], JerseyRead | None] = field(default=lambda f: None)
    absent: set[int] = field(default_factory=set)  # frames sin detección (fallo del detector u oclusión)
    team_visible: bool = True


@dataclass
class Scenario:
    frames: int
    players: list[SimPlayer]
    seed: int = 0
    dim: int = 64
    noise: float = 0.03
    keep_id_gap: int = 10  # hueco máximo en que el tracker conserva el ID (si es mayor, ID nuevo)
    swaps: list[tuple[int, str, str]] = field(default_factory=list)  # (frame, gt1, gt2): desde ese frame
    auto_swap_iou: float | None = None  # intercambia IDs aleatoriamente cuando dos cajas se superponen
    auto_swap_prob: float = 0.5


@dataclass
class Result:
    outputs: list[FrameOutput]
    truth: dict[int, dict[Box, str]]  # frame -> bbox -> gt

    def assignments(self) -> dict[str, dict[int, str]]:
        out: dict[str, dict[int, str]] = defaultdict(dict)
        for fo in self.outputs:
            for p in fo.players:
                gt = self.truth[fo.frame].get(tuple(p.bbox))  # type: ignore[arg-type]
                assert gt is not None, f"salida sin ground truth en frame {fo.frame}: {p}"
                out[gt][fo.frame] = p.player_id
        return out

    def ids(self, gt: str) -> list[str]:
        return list(dict.fromkeys(self.assignments()[gt].values()))

    def player(self, gt: str, frame: int) -> str | None:
        return self.assignments()[gt].get(frame)

    def out(self, frame: int) -> FrameOutput:
        return self.outputs[frame]

    def find(self, gt: str, frame: int) -> object | None:
        pid = self.player(gt, frame)
        return next((p for p in self.outputs[frame].players if p.player_id == pid), None)

    def frames_with_duplicate_ids(self) -> list[int]:
        return [fo.frame for fo in self.outputs if len({p.player_id for p in fo.players}) != len(fo.players)]

    def all_ids(self) -> set[str]:
        return {p.player_id for fo in self.outputs for p in fo.players}


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def run(scn: Scenario, manager: IdentityManager | None = None) -> Result:
    rng = np.random.default_rng(scn.seed)

    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    team_vec = {t: unit(rng.normal(size=scn.dim)) for t in Team}
    base = {p.gt: unit(0.6 * team_vec[p.team] + 0.8 * unit(rng.normal(size=scn.dim))) for p in scn.players}
    manager = manager or IdentityManager()

    next_id = 1
    track_of: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    swapped_pairs: set[frozenset[str]] = set()
    swaps_at: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for f, a, b in scn.swaps:
        swaps_at[f].append((a, b))

    outputs: list[FrameOutput] = []
    truth: dict[int, dict[Box, str]] = {}
    for f in range(scn.frames):
        visible: dict[str, Box] = {}
        for p in scn.players:
            c = p.path(f)
            if c is None or f in p.absent:
                continue
            w = 0.4 * p.height
            visible[p.gt] = (c[0] - w / 2, c[1] - p.height / 2, c[0] + w / 2, c[1] + p.height / 2)

        # IDs del tracker: conserva el ID en huecos cortos; ID nuevo tras huecos largos
        for gt in visible:
            if gt not in track_of or f - last_seen.get(gt, -(10**9)) > scn.keep_id_gap + 1:
                track_of[gt] = next_id
                next_id += 1
            last_seen[gt] = f
        for a, b in swaps_at.get(f, []):
            track_of[a], track_of[b] = track_of[b], track_of[a]
        if scn.auto_swap_iou is not None:
            gts = sorted(visible)
            for i, a in enumerate(gts):
                for b in gts[i + 1 :]:
                    pair = frozenset((a, b))
                    if pair in swapped_pairs or _iou(visible[a], visible[b]) < scn.auto_swap_iou:
                        continue
                    swapped_pairs.add(pair)
                    if rng.random() < scn.auto_swap_prob:
                        track_of[a], track_of[b] = track_of[b], track_of[a]
            # un par puede volver a intercambiarse en un cruce futuro
            for pair in list(swapped_pairs):
                a, b = tuple(pair)
                if a in visible and b in visible and _iou(visible[a], visible[b]) == 0.0:
                    swapped_pairs.discard(pair)

        obs = []
        truth[f] = {}
        by_gt = {p.gt: p for p in scn.players}
        for gt, box in visible.items():
            p = by_gt[gt]
            emb = unit(base[gt] + rng.normal(scale=scn.noise, size=scn.dim)).astype(np.float32)
            obs.append(
                Observation(
                    track_id=track_of[gt],
                    bbox=box,
                    confidence=0.9,
                    team=p.team if p.team_visible else None,
                    embedding=emb,
                    jersey=p.jersey(f),
                )
            )
            truth[f][box] = gt
        outputs.append(manager.update(f, obs))
    return Result(outputs=outputs, truth=truth)


def reads(
    number: int, conf: float = 0.95, every: int = 5, start: int = 0, end: int = 10**9
) -> Callable[[int], JerseyRead | None]:
    return lambda f: JerseyRead(number, conf) if start <= f <= end and (f - start) % every == 0 else None
