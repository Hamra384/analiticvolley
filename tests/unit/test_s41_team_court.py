"""S4.1 (#11): oficiales por color (AC-10), un líbero por equipo (AC-11), zona de juego (RF-2b)."""

import cv2
import numpy as np

from volley_cv.court import CourtMask
from volley_cv.identity.types import Team
from volley_cv.team import TeamClassifier

from .test_court import COURT, frame_with_court
from .test_team import DARK, TEAMS, WHITE, draw

GREY = (122, 126, 126)  # chaleco de los jueces de línea medido en A2


def test_ac10_line_judge_grey_is_official_not_red_libero() -> None:
    frame, boxes = draw([(GREY, 300, 600), (WHITE, 600, 600), (DARK, 900, 300)])
    clf = TeamClassifier(TEAMS, officials=[GREY])
    assert clf.is_official(frame, boxes) == [True, False, False]


def test_ac10_without_official_config_nothing_is_official() -> None:
    frame, boxes = draw([(GREY, 300, 600)])
    assert TeamClassifier(TEAMS).is_official(frame, boxes) == [False]


def test_ac11_at_most_one_libero_per_team_among_ambiguous() -> None:
    players = [(DARK, 200 + 150 * i, 300 + 30 * (i % 3)) for i in range(6)]  # franja de B
    players += [(WHITE, 1000, 325), (WHITE, 1100, 330), (WHITE, 1200, 320)]  # tres blancos en la franja de B
    frame, boxes = draw(players)
    teams = TeamClassifier(TEAMS).classify(frame, boxes)
    assert sum(t == Team.B for t in teams[6:]) <= 1


def test_play_zone_includes_free_zone_but_not_far_away() -> None:
    frame = frame_with_court()  # cancha en x 80..320, y 100..280 (alto 300)
    near_free = (40.0, 150.0, 60.0, 230.0)  # pies a 30 px de la cancha (zona libre)
    far_away = (0.0, 0.0, 10.0, 10.0)
    on_court = (150.0, 120.0, 170.0, 200.0)
    inside, play = CourtMask(COURT).zones(frame, [near_free, far_away, on_court], play_margin=0.12)
    assert inside == [False, False, True]
    assert play == [True, False, True]


def test_play_zone_excluded_region_still_excluded() -> None:
    frame = frame_with_court()
    mask = CourtMask(COURT, exclude_regions=[(0.0, 0.0, 1.0, 0.1)])
    _, play = mask.zones(frame, [(150.0, 0.0, 170.0, 20.0)], play_margin=0.12)
    assert play == [False]
    assert cv2.__version__  # (import usado)
    assert np.__version__
