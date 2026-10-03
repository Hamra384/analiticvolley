"""SPEC-004 AC-8: el clasificador de equipo informa si el torso es del líbero."""

from __future__ import annotations

from volley_cv.identity.types import Team
from volley_cv.team import TeamClassifier
from volley_cv.video_config import TeamColors

from .test_team import DARK, RED, TEAMS, WHITE, draw


def test_ac8_unambiguous_libero_color_and_main_colors() -> None:
    frame, boxes = draw([(RED, 300, 600), (DARK, 700, 300), (WHITE, 1000, 650)])
    assert TeamClassifier(TEAMS).classify_roles(frame, boxes) == [
        (Team.A, True),
        (Team.B, False),
        (Team.A, False),
    ]


def test_ac8_ambiguous_white_resolved_as_libero_by_side() -> None:
    # como test_ac4_ambiguous_white_resolved_by_side_of_dark_team: un blanco entre los oscuros es líbero de B
    players = [(DARK, 200 + 150 * i, 230 + 10 * i) for i in range(5)]
    players += [(WHITE, 1100, 250)]
    players += [(WHITE, 200 + 150 * i, 560 + 20 * i) for i in range(5)]
    frame, boxes = draw(players)
    roles = TeamClassifier(TEAMS).classify_roles(frame, boxes)
    assert roles[5] == (Team.B, True)
    assert roles[6:] == [(Team.A, False)] * 5


def test_ac8_libero_number_marks_libero() -> None:
    teams = {
        "A": TeamColors(name="Japón", main=WHITE, libero=RED),
        "B": TeamColors(name="Argentina", main=DARK, libero=WHITE, libero_numbers=[19]),
    }
    frame, boxes = draw([(WHITE, 300, 600)])
    assert TeamClassifier(teams).classify_roles(frame, boxes, numbers=[19]) == [(Team.B, True)]


def test_ac8_classify_keeps_returning_teams_only() -> None:
    frame, boxes = draw([(RED, 300, 600)])
    assert TeamClassifier(TEAMS).classify(frame, boxes) == [Team.A]
