"""Clasificación de equipo por color de torso con desambiguación de líberos por lado (SPEC-002 RF-3).

1. Color de torso = mediana Lab de la franja 20-45 % de la altura de la caja (60 % central del ancho).
2. Se compara contra los prototipos configurados (principal y líbero de cada equipo). Si el más cercano
   está a más de `max_dist` -> desconocido.
3. Si todos los prototipos a menos de `distancia mínima + ambiguity_margin` son del mismo equipo -> ese
   equipo, y el punto de apoyo del jugador se guarda como evidencia del lado de ese equipo en el tramo.
4. Si hay prototipos cercanos de los dos equipos (p. ej. líbero de B del mismo color que el principal de A),
   se decide con la evidencia de lado del tramo: dentro de la franja vertical ocupada por los jugadores no
   ambiguos de un equipo -> ese equipo. Sin evidencia -> desconocido (nunca se asume lado = equipo).
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
    ) -> None:
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
        for k, close in ambiguous:
            # caso tratable: el color es el principal de un equipo (X) y el líbero del otro (Y)
            liberos = {t for t, lib in close if lib}
            mains = {t for t, lib in close if not lib}
            if len(liberos) != 1 or len(mains - liberos) != 1:
                continue
            (owner,) = liberos
            (other,) = mains - liberos
            result[k] = self._by_side(boxes[k][3], owner, other, h)
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

    def _by_side(self, feet: float, owner: Team, other: Team, h: float) -> Team | None:
        """Equipo de un color ambiguo (principal de `other` = líbero de `owner`) según el lado en el tramo.

        Conservador: ante cualquier duda devuelve None (un equipo equivocado es peor que desconocido).
        """
        own = self._band(owner, h)
        if own is None:
            return None
        med, lo, hi = own
        rival = self._band(other, h)
        if rival is not None:
            if abs(rival[0] - med) < self.min_separation * h:
                return None  # los lados no están separados en la imagen
            d_own, d_rival = abs(feet - med), abs(feet - rival[0])
            if lo <= feet <= hi and d_own < d_rival:
                return owner
            if not lo <= feet <= hi and d_rival < d_own:
                return other
            return None
        if lo <= feet <= hi:
            # sin evidencia del otro equipo, solo el centro de la franja es suficientemente seguro
            return owner if abs(feet - med) <= (hi - lo) / 4 else None
        gap = (lo - feet) if feet < lo else (feet - hi)
        return other if gap >= self.clear_gap * h else None

    def _forget(self) -> None:
        for q in self._side.values():
            while q and q[0][0] <= self._frame - self.side_memory:
                q.popleft()

    def reset(self) -> None:
        """Corte de edición: se descarta la evidencia de lado del tramo anterior."""
        for q in self._side.values():
            q.clear()
