"""AC-3 y AC-4 (SPEC-002): equipo por color de torso; líbero ambiguo resuelto por lado o desconocido."""

import cv2
import numpy as np

from volley_cv.identity.types import Team
from volley_cv.team import TeamClassifier
from volley_cv.video_config import TeamColors

WHITE, DARK, RED, GREEN = (190, 130, 128), (60, 130, 124), (126, 162, 150), (120, 70, 180)
# Como jpn_arg_2026: el líbero de B es blanco, igual que el principal de A
TEAMS = {
    "A": TeamColors(name="Japón", main=WHITE, libero=RED),
    "B": TeamColors(name="Argentina", main=DARK, libero=WHITE),
}


def lab_to_bgr(lab: tuple[int, int, int]) -> tuple[int, int, int]:
    px = np.uint8([[lab]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])  # type: ignore[return-value]


def draw(
    players: list[tuple[tuple[int, int, int], float, float]],
) -> tuple[np.ndarray, list[tuple[float, ...]]]:
    """players: (color Lab, x, y_pies). Devuelve frame 1280x720 y cajas de 40x100."""
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    boxes = []
    for lab, x, feet in players:
        x1, y1, x2, y2 = x, feet - 100, x + 40, feet
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), lab_to_bgr(lab), -1)
        boxes.append((x1, y1, x2, y2))
    return frame, boxes


def test_ac3_main_colors_and_unambiguous_libero() -> None:
    darks = [(DARK, 300 + 120 * i, 250 + 8 * i) for i in range(4)]  # evidencia de lado de B (>= 4 apoyos)
    frame, boxes = draw([(WHITE, 100, 600), *darks, (RED, 500, 620), (GREEN, 900, 600)])
    teams = TeamClassifier(TEAMS).classify(frame, boxes)
    assert teams[1:5] == [Team.B] * 4  # azul oscuro = B
    assert teams[5] == Team.A  # rojo = líbero de A
    assert teams[6] is None  # color lejano a todos
    # el blanco es ambiguo (A principal / líbero B): está claramente fuera de la franja de B -> A (ver AC-4)
    assert teams[0] == Team.A


def test_ac3_single_rival_player_is_not_enough_side_evidence() -> None:
    frame, boxes = draw([(WHITE, 100, 600), (DARK, 300, 250)])
    assert TeamClassifier(TEAMS).classify(frame, boxes)[0] is None


def test_ac4_ambiguous_white_resolved_by_side_of_dark_team() -> None:
    # B (oscuros) del lado lejano (pies ~ y 230-280); un blanco entre ellos = líbero de B; blancos abajo = A
    players = [(DARK, 200 + 150 * i, 230 + 10 * i) for i in range(5)]
    players += [(WHITE, 1100, 250)]  # blanco del lado de B (sin superponerse a ningún oscuro)
    players += [(WHITE, 200 + 150 * i, 560 + 20 * i) for i in range(5)]  # blancos del lado cercano
    frame, boxes = draw(players)
    teams = TeamClassifier(TEAMS).classify(frame, boxes)
    assert teams[:5] == [Team.B] * 5
    assert teams[5] == Team.B
    assert teams[6:] == [Team.A] * 5


def test_ac4_uses_side_of_libero_owner_even_if_other_team_has_evidence() -> None:
    """Regresión S4 (A2 real): con evidencia de ambos equipos, un blanco fuera de la franja de B es de A."""
    players = [(DARK, 200 + 150 * i, 230 + 10 * i) for i in range(5)]  # B lejos
    players += [(RED, 400, 300)]  # líbero de A cerca de la red: evidencia de A en otra franja
    players += [(WHITE, 300, 620), (WHITE, 700, 650)]  # blancos del lado cercano
    frame, boxes = draw(players)
    teams = TeamClassifier(TEAMS).classify(frame, boxes)
    assert teams[5] == Team.A
    assert teams[6:] == [Team.A, Team.A]


def test_ac4_ambiguous_without_side_evidence_is_unknown() -> None:
    frame, boxes = draw([(WHITE, 200, 600), (WHITE, 500, 300)])
    assert TeamClassifier(TEAMS).classify(frame, boxes) == [None, None]


def test_side_evidence_is_discarded_on_reset() -> None:
    clf = TeamClassifier(TEAMS)
    frame, boxes = draw([(DARK, 200 + 150 * i, 250) for i in range(5)])
    clf.classify(frame, boxes)
    clf.reset()
    frame, boxes = draw([(WHITE, 640, 250)])
    assert clf.classify(frame, boxes) == [None]


def test_side_evidence_accumulates_across_frames_within_shot() -> None:
    clf = TeamClassifier(TEAMS)
    frame, boxes = draw([(DARK, 200 + 150 * i, 250) for i in range(5)])
    clf.classify(frame, boxes)
    frame, boxes = draw([(WHITE, 640, 255), (WHITE, 300, 600)])  # frame siguiente, sin oscuros visibles
    assert clf.classify(frame, boxes) == [Team.B, Team.A]
