"""SPEC-005 AC-1…AC-6: BallTracker con estados (trayectorias sintéticas)."""

from __future__ import annotations

import numpy as np

from volley_cv.ball import BallTracker, BallTrackerConfig

CFG = BallTrackerConfig()
D = 16.0  # diámetro de la pelota en px


def parabola(f: int) -> tuple[float, float]:
    """Pelota en vuelo: x constante en velocidad, y con gravedad (px/frame²)."""
    return 200.0 + 12.0 * f, 700.0 - 25.0 * f + 0.4 * f * f


def det(x: float, y: float, conf: float = 0.8) -> np.ndarray:
    return np.array([[x, y, D, D, conf]], dtype=np.float32)


NONE = np.zeros((0, 5), dtype=np.float32)


def test_ac1_first_detection_then_tracked() -> None:
    tr = BallTracker()
    outs = [tr.update(det(*parabola(f))) for f in range(20)]
    assert outs[0].state == "DETECTED"
    assert all(o.state == "TRACKED" for o in outs[1:])
    assert outs[-1].position is not None
    assert np.hypot(outs[-1].position[0] - parabola(19)[0], outs[-1].position[1] - parabola(19)[1]) < 1.0


def test_ac2_short_gap_is_predicted_close_to_truth() -> None:
    tr = BallTracker()
    for f in range(15):
        tr.update(det(*parabola(f)))
    gap = range(15, 15 + CFG.max_predict)
    outs = [tr.update(NONE) for _ in gap]
    assert all(o.state == "PREDICTED" for o in outs)
    for f, o in zip(gap, outs, strict=True):
        assert o.position is not None
        assert np.hypot(o.position[0] - parabola(f)[0], o.position[1] - parabola(f)[1]) < 2 * D
    assert outs[0].confidence > outs[-1].confidence  # la confianza baja mientras se predice
    nxt = 15 + CFG.max_predict
    assert tr.update(det(*parabola(nxt))).state == "TRACKED"


def test_ac3_long_gap_is_lost_then_reacquired() -> None:
    tr = BallTracker()
    for f in range(15):
        tr.update(det(*parabola(f)))
    outs = [tr.update(NONE) for _ in range(CFG.max_predict + 5)]
    assert outs[CFG.max_predict - 1].state == "PREDICTED"
    assert all(o.state == "LOST" and o.position is None for o in outs[CFG.max_predict :])
    f = 15 + CFG.max_predict + 5
    assert tr.update(det(*parabola(f))).state == "REACQUIRED"


def test_ac4_no_detection_is_never_detected_or_tracked() -> None:
    tr = BallTracker()
    rng = np.random.default_rng(0)
    for f in range(200):
        has = rng.random() < 0.5
        out = tr.update(det(*parabola(f % 40)) if has else NONE)
        if not has:
            assert out.state not in ("DETECTED", "TRACKED")


def test_ac5_far_false_positive_is_not_associated() -> None:
    tr = BallTracker()
    for f in range(15):
        tr.update(det(*parabola(f)))
    x, y = parabola(15)
    far = np.array([[1700.0, 100.0, D, D, 0.95]], dtype=np.float32)  # cartel LED, más confiable
    out = tr.update(np.vstack([far, det(x + 3, y - 2, 0.3)]))
    assert out.state == "TRACKED"
    assert out.position is not None and abs(out.position[0] - x) < D
    out = tr.update(far)  # solo el falso positivo: se predice, no se salta
    assert out.state == "PREDICTED"
    assert out.position is not None and out.position[0] < 1000


def test_ac6_reset_on_cut() -> None:
    tr = BallTracker()
    for f in range(10):
        tr.update(det(*parabola(f)))
    tr.reset()
    assert tr.update(NONE).state == "LOST"
    assert tr.update(det(500.0, 400.0)).state == "DETECTED"


def test_low_confidence_alone_does_not_start_a_track() -> None:
    tr = BallTracker()
    out = tr.update(det(500.0, 400.0, conf=CFG.start_conf / 2))
    assert out.state == "LOST" and out.position is None


def test_track_id_changes_after_a_cut() -> None:
    tr = BallTracker()
    a = tr.update(det(500.0, 400.0)).track_id
    tr.reset()
    b = tr.update(det(500.0, 400.0)).track_id
    assert a != b
