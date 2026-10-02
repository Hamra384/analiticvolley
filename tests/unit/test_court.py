"""AC-2 (SPEC-002): solo pasan las personas con el apoyo sobre la cancha y fuera de regiones excluidas."""

import cv2
import numpy as np

from volley_cv.court import CourtMask
from volley_cv.video_config import CourtConfig, HsvRange

ORANGE_HSV = (10, 150, 200)


def frame_with_court() -> np.ndarray:
    hsv = np.zeros((300, 400, 3), dtype=np.uint8)
    hsv[:] = (85, 120, 120)  # zona libre verde/turquesa
    hsv[100:280, 80:320] = ORANGE_HSV  # cancha naranja
    hsv[0:30, :] = ORANGE_HSV  # una franja naranja arriba (como un marcador): se excluye por config
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


COURT = CourtConfig(hsv_ranges=[HsvRange(lower=(3, 25, 80), upper=(18, 255, 255))], margin_px=5)


def test_ac2_feet_inside_court_pass_outside_do_not() -> None:
    mask = CourtMask(COURT)
    inside = (150.0, 120.0, 170.0, 200.0)  # pies en y=200, x=160: cancha
    outside = (20.0, 120.0, 40.0, 200.0)  # pies en x=30: zona libre
    assert mask.in_court(frame_with_court(), [inside, outside]) == [True, False]


def test_ac2_margin_tolerates_feet_just_outside_the_line() -> None:
    mask = CourtMask(COURT)
    near = (60.0, 150.0, 76.0, 230.0)  # pies en x=68: 12 px fuera, margen 5 -> afuera
    assert mask.in_court(frame_with_court(), [near]) == [False]
    wide = CourtMask(CourtConfig(hsv_ranges=COURT.hsv_ranges, margin_px=20))
    assert wide.in_court(frame_with_court(), [near]) == [True]


def test_ac2_excluded_region_never_passes() -> None:
    on_court_top = (150.0, 30.0, 170.0, 110.0)  # pies en y=110: cancha (empieza en y=100)
    assert CourtMask(COURT).in_court(frame_with_court(), [on_court_top]) == [True]
    mask = CourtMask(COURT, exclude_regions=[(0.0, 0.0, 1.0, 0.4)])  # excluye y < 120, aunque sea cancha
    assert mask.in_court(frame_with_court(), [on_court_top]) == [False]


def test_disconnected_court_colored_blobs_are_ignored() -> None:
    """Regresión S4 (A2 real): público con ropa naranja o la silla del árbitro no son la cancha."""
    frame = frame_with_court()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[40:70, 330:380] = ORANGE_HSV  # mancha naranja separada de la cancha
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    on_blob = (345.0, 0.0, 365.0, 60.0)
    on_court = (150.0, 120.0, 170.0, 200.0)
    assert CourtMask(COURT, exclude_regions=[(0.0, 0.0, 1.0, 0.1)]).in_court(frame, [on_blob, on_court]) == [
        False,
        True,
    ]


def test_white_court_lines_do_not_split_the_court() -> None:
    """Las líneas de ataque/centro (blancas, finas) no deben dejar zonas de la cancha afuera."""
    frame = frame_with_court()
    # 3 px a 300 px de alto ~ 10 px a 1080p (las líneas reales miden 6-10 px a 1080p)
    frame[150:153, 80:320] = (255, 255, 255)  # línea de ataque
    frame[200:203, 80:320] = (255, 255, 255)  # línea central
    zones = [(150.0, 100.0, 170.0, 140.0), (150.0, 120.0, 170.0, 180.0), (150.0, 160.0, 170.0, 260.0)]
    assert CourtMask(COURT).in_court(frame, zones) == [True, True, True]


def test_feet_outside_image_do_not_pass() -> None:
    assert CourtMask(COURT).in_court(frame_with_court(), [(150.0, 250.0, 170.0, 400.0)]) == [False]
