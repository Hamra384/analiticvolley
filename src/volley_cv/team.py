"""Clasificación de equipo por color de torso con desambiguación de líberos por lado (SPEC-002 RF-3).

1. Color de torso = mediana Lab de la franja 20-45 % de la altura de la caja (60 % central del ancho).
2. Se compara contra los prototipos configurados (principal y líbero de cada equipo). Si el más cercano
   está a más de `max_dist` -> desconocido.
3. Si todos los prototipos a menos de `distancia mínima + ambiguity_margin` son del mismo equipo -> ese
   equipo, y el punto de apoyo del jugador se guarda como evidencia del lado de ese equipo en el tramo.
4. Si hay prototipos cercanos de los dos equipos (p. ej. líbero de B del mismo color que el principal de A),
   por defecto es el equipo de color principal (A: ~6 jugadores contra un líbero, RF-3d); es el líbero de B
   solo con evidencia de lado del tramo (ver `_by_side`). No se asume lado = equipo sin esa evidencia.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from volley_cv.identity.types import Team
from volley_cv.video_config import TeamColors

Box = tuple[float, float, float, float]


def torso_lab(frame: NDArray[np.uint8], box: Box) -> NDArray[np.float64] | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ty1, ty2 = int(y1 + 0.2 * h), int(y1 + 0.45 * h)
    tx1, tx2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    crop = frame[max(ty1, 0) : max(ty2, 0), max(tx1, 0) : max(tx2, 0)]
    if crop.size == 0:
        return None
    lab = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB), dtype=np.float64).reshape(-1, 3)
    return np.asarray(np.median(lab, axis=0), dtype=np.float64)


class TeamClassifier:
    def __init__(
        self,
        teams: dict[str, TeamColors],
        max_dist: float = 45.0,
        ambiguity_margin: float = 10.0,
        side_memory: int = 90,
        side_margin: float = 0.02,
        min_evidence: int = 4,
        max_spread: float = 0.30,
        min_separation: float = 0.08,
        clear_gap: float = 0.05,
        officials: Sequence[tuple[float, float, float]] = (),
    ) -> None:
        self._officials = [np.array(o, dtype=np.float64) for o in officials]
        self.max_dist, self.ambiguity_margin = max_dist, ambiguity_margin
        self.side_memory, self.side_margin = side_memory, side_margin
        # evidencia de lado (fracciones del alto del frame): mínimo de apoyos, dispersión máxima (IQR) para
        # considerarla compacta, separación mínima entre equipos y distancia mínima fuera de la franja
        self.min_evidence, self.max_spread = min_evidence, max_spread
        self.min_separation, self.clear_gap = min_separation, clear_gap
        # (equipo, es_libero, color Lab)
        self._protos: list[tuple[Team, bool, NDArray[np.float64]]] = []
        for key, colors in sorted(teams.items()):
            team = Team(key)
            self._protos.append((team, False, np.array(colors.main, dtype=np.float64)))
            if colors.libero is not None:
                self._protos.append((team, True, np.array(colors.libero, dtype=np.float64)))
        self._frame = 0
        self._side: dict[Team, deque[tuple[int, float]]] = {t: deque() for t in Team}

    def classify(self, frame: NDArray[np.uint8], boxes: Sequence[Box]) -> list[Team | None]:
        self._frame += 1
        h = frame.shape[0]
        result: list[Team | None] = [None] * len(boxes)
        ambiguous: list[tuple[int, set[tuple[Team, bool]]]] = []
        for k, box in enumerate(boxes):
            lab = torso_lab(frame, box)
            if lab is None:
                continue
            dists = [(float(np.linalg.norm(lab - p)), t, lib) for t, lib, p in self._protos]
            d_min = min(d for d, _, _ in dists)
            if d_min > self.max_dist:
                continue
            close = {(t, lib) for d, t, lib in dists if d <= d_min + self.ambiguity_margin}
            if len({t for t, _ in close}) == 1:
                team = next(iter(close))[0]
                result[k] = team
                self._side[team].append((self._frame, box[3]))
            else:
                ambiguous.append((k, close))
        self._forget()
        # caso tratable: el color es el principal de un equipo (X = other) y el líbero del otro (Y = owner)
        pairs: dict[int, tuple[Team, Team]] = {}
        for k, close in ambiguous:
            liberos = {t for t, lib in close if lib}
            mains = {t for t, lib in close if not lib}
            if len(liberos) == 1 and len(mains - liberos) == 1:
                pairs[k] = (next(iter(liberos)), next(iter(mains - liberos)))
        # lado de X cuando sus jugadores son todos ambiguos (ARG: blanco = Japón y líbero argentino): la
        # mediana de los ambiguos del frame, que son casi todos de X (a lo sumo uno es el líbero)
        fallback: dict[tuple[Team, Team], float] = {}
        for pair in set(pairs.values()):
            feet = [boxes[k][3] for k, p in pairs.items() if p == pair]
            if len(feet) >= 3:
                fallback[pair] = float(np.median(feet))
        as_libero: dict[Team, list[tuple[int, Team, float]]] = {}
        for k, (owner, other) in pairs.items():
            rival = self._band(other, h)
            rival_med = rival[0] if rival else fallback.get((owner, other))
            result[k] = self._by_side(boxes[k][3], owner, other, h, rival_med)
            if result[k] == owner:
                as_libero.setdefault(owner, []).append(
                    (k, other, rival_med if rival_med is not None else 0.0)
                )
        # RF-3c: a lo sumo un líbero por equipo en cancha. Si varios se resolvieron como líbero, queda el más
        # alejado del lado rival (el líbero suele ser el más retrasado); el resto vuelve al equipo de color
        # principal (RF-3d), no a "desconocido"
        for cands in as_libero.values():
            if len(cands) > 1:
                keep = max(cands, key=lambda c: abs(boxes[c[0]][3] - c[2]))[0]
                for k, other, _ in cands:
                    if k != keep:
                        result[k] = other
        return result

    def _band(self, team: Team, h: float) -> tuple[float, float, float] | None:
        """(mediana, límite inferior, límite superior) robustos del apoyo de `team` en el tramo, o None si la
        evidencia no alcanza o no es compacta (p. ej. plano lateral: el equipo ocupa toda la altura)."""
        ys = [y for _, y in self._side[team]]
        if len(ys) < self.min_evidence:
            return None
        q1, med, q3 = (float(v) for v in np.percentile(ys, [25, 50, 75]))
        if q3 - q1 > self.max_spread * h:
            return None
        half = max((q3 - q1) / 2, self.side_margin * h)
        return med, med - half - self.side_margin * h, med + half + self.side_margin * h

    def _by_side(self, feet: float, owner: Team, other: Team, h: float, rival_med: float | None) -> Team:
        """Equipo de un color ambiguo (principal de `other` = líbero de `owner`) según el lado en el tramo.

        RF-3d (S4.1): prior por cantidad. Con ese color hay ~6 jugadores de `other` y a lo sumo un líbero de
        `owner`, así que por defecto es `other`; solo con evidencia de lado es el líbero de `owner`:
        dentro de la franja de `owner` y más cerca de ella que del rival, o **más allá** de esa franja del
        lado opuesto al rival (el líbero suele ser el más retrasado de su equipo; revisión S4.1, H1).
        """
        own = self._band(owner, h)
        if own is None:
            return other
        med, lo, hi = own
        if rival_med is None:
            # sin saber dónde está el rival, solo el centro de la franja es suficientemente seguro
            return owner if lo <= feet <= hi and abs(feet - med) <= (hi - lo) / 4 else other
        if abs(rival_med - med) < self.min_separation * h:
            return other  # los lados no están separados en la imagen
        if lo <= feet <= hi:
            return owner if abs(feet - med) < abs(feet - rival_med) else other
        beyond_own_side = (feet - med) * (med - rival_med) > 0
        return owner if beyond_own_side else other

    def _forget(self) -> None:
        for q in self._side.values():
            while q and q[0][0] <= self._frame - self.side_memory:
                q.popleft()

    def is_official(self, frame: NDArray[np.uint8], boxes: Sequence[Box]) -> list[bool]:
        """True si el prototipo de color más cercano al torso es de un oficial (RF-3b)."""
        if not self._officials:
            return [False] * len(boxes)
        out = []
        for box in boxes:
            lab = torso_lab(frame, box)
            if lab is None:
                out.append(False)
                continue
            d_off = min(float(np.linalg.norm(lab - p)) for p in self._officials)
            d_team = min(float(np.linalg.norm(lab - p)) for _, _, p in self._protos)
            out.append(d_off < d_team and d_off <= self.max_dist)
        return out

    def reset(self) -> None:
        """Corte de edición: se descarta la evidencia de lado del tramo anterior."""
        for q in self._side.values():
            q.clear()
