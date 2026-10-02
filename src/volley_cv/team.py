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
        side_margin: float = 0.04,
    ) -> None:
        self.max_dist, self.ambiguity_margin = max_dist, ambiguity_margin
        self.side_memory, self.side_margin = side_memory, side_margin
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
        margin = self.side_margin * h
        for k, close in ambiguous:
            # caso tratable: el color es el principal de un equipo (X) y el líbero del otro (Y). Decide la
            # franja de Y (de sus jugadores no ambiguos): dentro -> líbero de Y; fuera -> X. Sin evidencia de
            # Y, o cualquier otra combinación ambigua -> desconocido.
            liberos = {t for t, lib in close if lib}
            mains = {t for t, lib in close if not lib}
            if len(liberos) != 1 or len(mains - liberos) != 1:
                continue
            (owner,) = liberos
            (other,) = mains - liberos
            ys = [y for _, y in self._side[owner]]
            if not ys:
                continue
            lo, hi = min(ys) - margin, max(ys) + margin
            result[k] = owner if lo <= boxes[k][3] <= hi else other
        return result

    def _forget(self) -> None:
        for q in self._side.values():
            while q and q[0][0] <= self._frame - self.side_memory:
                q.popleft()

    def reset(self) -> None:
        """Corte de edición: se descarta la evidencia de lado del tramo anterior."""
        for q in self._side.values():
            q.clear()
