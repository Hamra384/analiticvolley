"""AC-1 (SPEC-002): el ShotDetector marca el corte y no marca un paneo suave."""

import numpy as np

from volley_cv.video.shots import ShotDetector


def scene(seed: int, w: int = 640, h: int = 360) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 20, w // 20, 3), dtype=np.uint8)
    return np.kron(base, np.ones((20, 20, 1), dtype=np.uint8))  # bloques grandes: textura de "escena"


def test_ac1_cut_is_detected_exactly_once() -> None:
    a, b = scene(1), scene(2)
    det = ShotDetector()
    cuts = []
    for i in range(120):
        src = a if i < 70 else b
        frame = np.roll(src, shift=i % 70, axis=1)  # paneo de 1 px por frame
        if det.update(frame):
            cuts.append(i)
    assert cuts == [70]


def test_ac1_smooth_pan_has_no_cuts() -> None:
    a = scene(3)
    det = ShotDetector()
    assert not any(det.update(np.roll(a, shift=2 * i, axis=1)) for i in range(150))


def test_reset_forgets_previous_frame() -> None:
    det = ShotDetector()
    det.update(scene(5))
    det.reset()
    assert det.update(scene(6)) is False  # tras reset, el primer frame nunca es corte


def test_first_frame_is_not_a_cut() -> None:
    assert ShotDetector().update(scene(4)) is False
