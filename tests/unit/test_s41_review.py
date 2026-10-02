"""Regresiones de la revisión independiente de S4.1 (H1, M3) sobre TeamClassifier."""

from volley_cv.identity.types import Team
from volley_cv.team import TeamClassifier

from .test_team import DARK, TEAMS, WHITE, draw
from .test_team_regressions import KOR, RED_JP


def test_h1_kor_deep_libero_behind_own_team_stays_with_own_team() -> None:
    """Japón (rojos) cerca, pies 0,60-0,80 del alto; Corea lejos 0,30-0,45; líbero blanco de Japón a 0,95."""
    h = 720
    players = [(RED_JP, 150 + 180 * i, int(h * (0.60 + 0.04 * i))) for i in range(6)]
    players += [(WHITE, 200 + 200 * i, int(h * (0.30 + 0.03 * i))) for i in range(5)]  # coreanos
    players += [(WHITE, 1150, int(h * 0.95))]  # líbero de Japón, el más retrasado
    frame, boxes = draw(players)
    teams = TeamClassifier(KOR).classify(frame, boxes)
    assert teams[-1] == Team.A
    assert teams[6:11] == [Team.B] * 5


def test_h1_arg_deep_libero_uses_ambiguous_majority_as_rival_side() -> None:
    """ARG: los blancos son casi todos de Japón (cerca); el líbero argentino blanco detrás de los oscuros."""
    h = 720
    players = [(DARK, 200 + 180 * i, int(h * (0.38 + 0.02 * i))) for i in range(5)]  # Argentina lejos
    players += [(WHITE, 150 + 200 * i, int(h * (0.70 + 0.03 * i))) for i in range(5)]  # Japón cerca
    players += [(WHITE, 1150, int(h * 0.22))]  # líbero argentino, el más lejano
    frame, boxes = draw(players)
    teams = TeamClassifier(TEAMS).classify(frame, boxes)
    assert teams[-1] == Team.B
    assert teams[5:10] == [Team.A] * 5


def test_m3_extra_libero_candidates_go_to_main_team_not_unknown() -> None:
    players = [(DARK, 200 + 150 * i, 300 + 30 * (i % 3)) for i in range(6)]  # franja de B
    players += [(WHITE, 1000, 325), (WHITE, 1100, 330), (WHITE, 1200, 320)]
    frame, boxes = draw(players)
    teams = TeamClassifier(TEAMS).classify(frame, boxes)[6:]
    assert sum(t == Team.B for t in teams) <= 1
    assert None not in teams  # RF-3d: los descartados como líbero vuelven al equipo de color principal
