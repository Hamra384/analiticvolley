"""Video de debug (punto 14 del MVP; SPEC-002 RF-6).

Caja por jugador con el color de su equipo, etiqueta corta (A04 #7), estado si no es TRACKED, track_id
opcional, trayectoria de los pies y una barra con el frame y los cortes de edición. Todo se puede apagar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from volley_cv.output.schema import FrameOutput

TEAM_COLORS = {"TEAM_A": (255, 170, 0), "TEAM_B": (40, 60, 255)}  # BGR: celeste / rojo
STATE_TAGS = {"DETECTED": "NEW", "REIDENTIFIED": "REID", "OCCLUDED": "OCC", "LOST": "LOST", "TRACKED": ""}


def short_id(player_id: str) -> str:
    """TEAM_A_PLAYER_04 -> A04."""
    parts = player_id.split("_")
    return f"{parts[1]}{parts[3]}" if len(parts) == 4 else player_id


BALL_COLOR = (0, 255, 255)  # amarillo (BGR)


@dataclass(frozen=True)
class DebugOptions:
    boxes: bool = True
    labels: bool = True
    track_ids: bool = False
    jersey: bool = True
    state: bool = True
    trails: bool = True
    hud: bool = True
    trail_frames: int = 30


class DebugRenderer:
    def __init__(self, options: DebugOptions | None = None) -> None:
        self.options = options or DebugOptions()
        self._trails: dict[str, deque[tuple[int, int]]] = {}
        self._last_seen: dict[str, int] = {}

    def draw(self, frame: NDArray[np.uint8], out: FrameOutput, cut: bool = False) -> NDArray[np.uint8]:
        o = self.options
        img = frame.copy()
        for p in out.players:
            color = TEAM_COLORS[p.team_id]
            x1, y1, x2, y2 = (round(v) for v in p.bbox)
            feet = ((x1 + x2) // 2, y2)
            trail = self._trails.setdefault(p.player_id, deque(maxlen=o.trail_frames))
            if out.frame - self._last_seen.get(p.player_id, out.frame - 1) > 1:
                trail.clear()  # desapareció: no unir la posición vieja con la nueva
            self._last_seen[p.player_id] = out.frame
            trail.append(feet)
            if o.trails and len(trail) > 1:
                cv2.polylines(img, [np.array(trail, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            if o.boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = []
            if o.labels:
                text.append(short_id(p.player_id))
            if o.jersey and p.jersey_number is not None:
                text.append(f"#{p.jersey_number}")
            if o.state and STATE_TAGS.get(p.state):
                text.append(STATE_TAGS[p.state])
            if o.track_ids:
                text.append(p.track_id.replace("track_", "t"))
            if text:
                label = " ".join(text)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ty = max(y1 - 6, th + 2)
                cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 3), (0, 0, 0), -1)
                cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        b = out.ball
        if b is not None and b.position is not None:
            # SPEC-005 RF-6: detectada = círculo lleno; predicha = solo contorno (nunca parece detectada)
            c = (round(b.position[0]), round(b.position[1]))
            filled = b.state in ("DETECTED", "TRACKED", "REACQUIRED")
            cv2.circle(img, c, 9, BALL_COLOR, -1 if filled else 2, cv2.LINE_AA)
            cv2.circle(img, c, 13, (0, 0, 0), 1, cv2.LINE_AA)
        if o.hud:
            hud = f"frame {out.frame}  jugadores {len(out.players)}" + ("  CORTE" if cut else "")
            if b is not None:
                hud += f"  pelota {b.state}"
            cv2.rectangle(img, (0, 0), (520, 28), (0, 0, 0), -1)
            cv2.putText(img, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def reset(self) -> None:
        """Corte: se borran las trayectorias."""
        self._trails.clear()
        self._last_seen.clear()
