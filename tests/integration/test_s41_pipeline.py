"""S4.1 (#11) AC-9 y AC-10 de SPEC-002 en el pipeline: zona de juego con histéresis y oficiales excluidos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from volley_cv.court import CourtMask
from volley_cv.pipeline import Pipeline
from volley_cv.team import TeamClassifier

from ..unit.test_team import DARK, TEAMS  # con líbero rojo de A: el gris del juez queda a distancia 43
from .test_pipeline import COURT, WHITE, H, W, lab_to_bgr

GREY = (122, 126, 126)
N = 90


def path(i: int) -> tuple[float, float]:
    """Jugador que sale de la cancha (x < 100) hasta x = 70 entre los frames 20 y 60 y vuelve."""
    if i < 20 or i >= 60:
        return 200.0, 300.0
    t = (i - 20) / 40
    x = 200 - 130 * (1 - abs(2 * t - 1))  # 200 -> 70 -> 200
    return x, 300.0


PEOPLE = {
    "player": lambda i: path(i),
    "official": lambda i: (300.0, 200.0),  # gris, parado en la cancha (como un juez que pisa la línea)
    "outsider": lambda i: (70.0, 150.0),  # blanco, siempre en la zona libre: nunca pisa la cancha
}
# jugador oscuro (principal de B, no ambiguo): el test aísla la zona de juego, no la regla de líbero
COLORS = {"player": DARK, "official": GREY, "outsider": WHITE}


def boxes_at(i: int) -> dict[str, tuple[float, float, float, float]]:
    return {k: (p(i)[0], p(i)[1] - 80, p(i)[0] + 30, p(i)[1]) for k, p in PEOPLE.items()}


def make_frame(i: int) -> np.ndarray:
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[:] = (85, 120, 120)
    hsv[100:340, 100:560] = (10, 150, 200)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for k, (x1, y1, x2, y2) in boxes_at(i).items():
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), lab_to_bgr(COLORS[k]), -1)
    return frame


def who(box: Sequence[float], i: int) -> str:
    return min(boxes_at(i).items(), key=lambda kv: abs(kv[1][0] - box[0]) + abs(kv[1][3] - box[3]))[0]


class Det:
    def __init__(self) -> None:
        self.i = 0

    def detect(self, frame: np.ndarray) -> np.ndarray:
        d = np.array([[*b, 0.9] for b in boxes_at(self.i).values()], dtype=np.float32)
        self.i += 1
        return d


class Trk:
    def __init__(self, det: Det) -> None:
        self.det = det

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        i = self.det.i - 1
        ids = {"player": 1, "official": 2, "outsider": 3}
        return np.array([[*d[:4], ids[who(d, i)], d[4]] for d in detections], dtype=np.float32).reshape(-1, 6)

    def reset(self) -> None:
        pass


class Emb:
    def __init__(self, det: Det) -> None:
        self.det = det

    def embed(
        self, frame: np.ndarray, bxs: Sequence[tuple[float, float, float, float]]
    ) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        for b in bxs:
            v = np.zeros(8, dtype=np.float32)
            v[["player", "official", "outsider"].index(who(b, self.det.i - 1))] = 1.0
            out.append(v)
        return out


def run(tmp_path: Path) -> list[dict[str, str]]:
    import json

    det = Det()
    pipe = Pipeline(
        detector=det,
        tracker=Trk(det),
        embedder=Emb(det),
        court=CourtMask(COURT),
        teams=TeamClassifier(TEAMS, officials=[GREY]),
        play_margin=0.12,
    )
    pipe.run(((i, make_frame(i)) for i in range(N)), tmp_path, write_video=False)
    rows = [json.loads(x) for x in (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    return [{p["track_id"]: p["player_id"] for p in r["players"]} for r in rows]


def test_ac9_player_followed_off_court_keeps_identity(tmp_path: Path) -> None:
    frames = run(tmp_path)
    ids = [f.get("track_1") for f in frames[10:]]
    assert None not in ids  # nunca se pierde, ni al salir de la cancha
    assert len(set(ids)) == 1


def test_ac9_person_never_on_court_gets_no_identity(tmp_path: Path) -> None:
    assert all("track_3" not in f for f in run(tmp_path))


def test_ac10_official_never_reaches_identity(tmp_path: Path) -> None:
    assert all("track_2" not in f for f in run(tmp_path))
