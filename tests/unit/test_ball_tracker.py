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


def test_rf2b_static_decoy_does_not_capture_the_track() -> None:
    # A2 real: un alcanzapelotas sostiene una pelota de repuesto quieta fuera de la cancha (siempre visible)
    tr = BallTracker()
    decoy = np.array([[100.0, 100.0, D, D, 0.9]], dtype=np.float32)
    for _ in range(CFG.static_frames + 5):
        tr.update(decoy)
    outs = []
    for f in range(40):  # la pelota del juego aparece en vuelo (detectada 2 de cada 3 frames)
        live = det(*parabola(f), conf=0.5) if f % 3 else np.zeros((0, 5), dtype=np.float32)
        outs.append(tr.update(np.vstack([decoy, live])))
    for o in outs[5:]:
        assert o.position is not None
        assert np.hypot(o.position[0] - 100.0, o.position[1] - 100.0) > 3 * D  # nunca en el señuelo


def test_rf2b_decoy_alone_is_eventually_not_reported() -> None:
    tr = BallTracker()
    decoy = np.array([[100.0, 100.0, D, D, 0.9]], dtype=np.float32)
    outs = [tr.update(decoy) for _ in range(CFG.static_frames + 20)]
    assert outs[-1].state == "LOST" and outs[-1].position is None


def test_rf2b_slow_but_moving_ball_is_not_a_decoy() -> None:
    tr = BallTracker()
    outs = [tr.update(det(300.0 + 2.0 * f, 400.0)) for f in range(CFG.static_frames + 20)]
    assert outs[-1].state == "TRACKED"


# ── revisión independiente S5b ──────────────────────────────────────────────


def _lose(tr: BallTracker, gap: int) -> None:
    for f in range(15):
        tr.update(det(*parabola(f)))
    for _ in range(gap):
        tr.update(NONE)


def test_review_m1_low_confidence_far_candidate_after_lost_is_not_reacquired() -> None:
    tr = BallTracker()
    _lose(tr, 30)
    out = tr.update(det(1090.0, 385.0, conf=0.06))
    assert out.state == "LOST" and out.position is None


def test_review_m1_confident_far_detection_after_lost_starts_a_new_track() -> None:
    tr = BallTracker()
    _lose(tr, CFG.max_predict + 3)
    first = tr.update(NONE).track_id
    out = tr.update(det(1700.0, 150.0, conf=0.9))
    assert out.state == "DETECTED" and out.track_id != first


def test_review_m1_lost_prediction_does_not_drift_forever() -> None:
    tr = BallTracker()
    _lose(tr, 200)
    x, y = parabola(14)
    out = tr.update(det(x + 30, y - 20, conf=0.9))  # vuelve cerca de donde se perdió
    assert out.state in ("REACQUIRED", "DETECTED")
    assert out.position is not None and abs(out.position[0] - (x + 30)) < D


def test_review_m2_slow_moving_ball_is_not_a_decoy() -> None:
    tr = BallTracker()
    # 0,1 diámetros por frame: recorre 4,5 diámetros en 45 frames (no está quieta)
    outs = [tr.update(det(300.0 + 0.1 * D * f, 400.0)) for f in range(CFG.static_frames + 30)]
    assert all(o.state in ("DETECTED", "TRACKED") for o in outs)


def test_review_b3_two_candidates_on_the_same_spot_count_once() -> None:
    tr = BallTracker()
    pair = np.array([[100.0, 100.0, D, D, 0.9], [101.0, 101.0, D, D, 0.5]], dtype=np.float32)
    outs = [tr.update(pair) for _ in range(CFG.static_frames - 5)]
    assert outs[-1].state == "TRACKED"  # todavía no es señuelo
