"""Regresiones de la revisión independiente de S4 sobre TeamClassifier (hallazgos 1-4)."""

from volley_cv.identity.types import Team
from volley_cv.team import TeamClassifier
from volley_cv.video_config import TeamColors

from .test_team import DARK, RED, TEAMS, WHITE, draw

BLUE = (50, 135, 114)
# Orientación KOR: el líbero de A es blanco, igual que el principal de B
KOR = {
    "A": TeamColors(name="Japón", main=(72, 164, 131), libero=WHITE),
    "B": TeamColors(name="Corea", main=WHITE, libero=BLUE),
}
RED_JP = (72, 164, 131)


def test_finding1_white_near_net_between_teams_is_unknown_not_rival() -> None:
    """Oscuros (B) con pies en y 440-500; un blanco en y=525 (bloqueador japonés en la red) no es B."""
    players = [(DARK, 200 + 150 * i, 440 + 12 * i) for i in range(6)]
    players += [(WHITE, 640, 525)]
    frame, boxes = draw(players)
    assert TeamClassifier(TEAMS).classify(frame, boxes)[6] is None


def test_finding1_white_deep_in_rival_court_is_their_libero() -> None:
    players = [(DARK, 200 + 150 * i, 300 + 30 * (i % 3)) for i in range(6)]
    players += [(WHITE, 640, 330)]  # en el medio de la franja de B
    frame, boxes = draw(players)
    assert TeamClassifier(TEAMS).classify(frame, boxes)[6] == Team.B


def test_finding2_single_dark_outlier_does_not_flip_near_whites() -> None:
    players = [(DARK, 200 + 150 * i, 240 + 10 * i) for i in range(5)]
    players += [(DARK, 1100, 690)]  # un oficial oscuro pisando la cancha cercana
    players += [(WHITE, 150 + 200 * i, 600 + 15 * i) for i in range(5)]
    frame, boxes = draw(players)
    teams = TeamClassifier(TEAMS).classify(frame, boxes)
    assert teams[6:] == [Team.A] * 5


def test_finding3_kor_orientation_whites_are_korea_and_white_libero_in_red_band_is_japan() -> None:
    players = [(RED_JP, 200 + 150 * i, 560 + 15 * (i % 3)) for i in range(5)]  # Japón cerca
    players += [(WHITE, 640, 575)]  # líbero blanco de Japón, dentro de la franja roja
    players += [(WHITE, 200 + 200 * i, 250 + 10 * i) for i in range(5)]  # coreanos lejos
    frame, boxes = draw(players)
    teams = TeamClassifier(KOR).classify(frame, boxes)
    assert teams[5] == Team.A
    assert teams[6:] == [Team.B] * 5


def test_finding3_kor_without_side_evidence_is_unknown() -> None:
    frame, boxes = draw([(WHITE, 200 + 200 * i, 250 + 10 * i) for i in range(5)])
    assert TeamClassifier(KOR).classify(frame, boxes) == [None] * 5


def test_finding4_lateral_view_without_compact_side_is_unknown() -> None:
    """Plano lateral: los oscuros ocupan toda la altura (separados en x, no en y) -> no se decide por y."""
    players = [(DARK, 100 + 60 * i, 150 + 110 * i) for i in range(6)]  # pies de y=150 a y=700
    players += [(WHITE, 900 + 50 * i, 200 + 100 * i) for i in range(5)]
    frame, boxes = draw(players)
    assert TeamClassifier(TEAMS).classify(frame, boxes)[6:] == [None] * 5


def _both_sides(white_feet: float, a_feet: float = 620.0, b_feet: float = 260.0) -> Team | None:
    """Evidencia de ambos equipos (4 rojos = A, 4 oscuros = B) y un blanco ambiguo en `white_feet`."""
    players = [(RED, 150 + 120 * i, a_feet + 6 * i) for i in range(4)]
    players += [(DARK, 700 + 120 * i, b_feet + 6 * i) for i in range(4)]
    players += [(WHITE, 1180, white_feet)]
    frame, boxes = draw(players)
    return TeamClassifier(TEAMS).classify(frame, boxes)[-1]


def test_both_sides_white_inside_rival_band_is_their_libero() -> None:
    assert _both_sides(white_feet=268) == Team.B


def test_both_sides_white_near_own_team_is_own() -> None:
    assert _both_sides(white_feet=640) == Team.A


def test_both_sides_white_between_teams_is_unknown() -> None:
    assert _both_sides(white_feet=440) is None


def test_both_sides_not_separated_is_unknown() -> None:
    assert _both_sides(white_feet=300, a_feet=300, b_feet=280) is None


def test_owner_only_white_at_edge_of_band_is_unknown() -> None:
    """Solo evidencia de B: un blanco en el borde de la franja (no en el centro) no se decide."""
    players = [(DARK, 200 + 150 * i, 300 + 4 * i) for i in range(6)]
    players += [(WHITE, 1150, 300 + 4 * 2.5 + 0.035 * 720)]  # dentro de la franja, lejos del centro
    frame, boxes = draw(players)
    assert TeamClassifier(TEAMS).classify(frame, boxes)[-1] is None


def test_unambiguous_red_libero_still_classified() -> None:
    frame, boxes = draw([(RED, 400, 600)])
    assert TeamClassifier(TEAMS).classify(frame, boxes) == [Team.A]
