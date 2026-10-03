"""SPEC-005 AC-8 y RF-7: tramos de entrenamiento lejos de los clips de evaluación y armado de etiquetas."""

from __future__ import annotations

from tools.ball_dataset import training_windows
from volley_cv.config import PROJECT_ROOT, load_clips

CATALOG = load_clips(PROJECT_ROOT / "configs" / "eval" / "clips.yaml")
DURATIONS = {"jpn_arg_2026": 50.0 * 60, "jpn_kor_2026": 47.4 * 60}


def test_ac8_no_window_within_margin_of_an_eval_clip() -> None:
    wins = training_windows(CATALOG, DURATIONS, margin_s=60.0, window_s=2.0, every_s=20.0)
    assert wins  # hay material
    for video, start, end in wins:
        assert 0 <= start < end <= DURATIONS[video]
        for clip in CATALOG.clips:
            if clip.video == video:
                assert end <= clip.start_s - 60.0 or start >= clip.end_s + 60.0, (video, start, clip.id)


def test_windows_are_spread_over_both_videos() -> None:
    wins = training_windows(CATALOG, DURATIONS, margin_s=60.0, window_s=2.0, every_s=20.0)
    assert {v for v, _, _ in wins} == set(DURATIONS)
    assert len(wins) > 200  # ~97 min / 20 s, menos las zonas excluidas


def _d(x: float, conf: float = 0.5) -> tuple[float, float, float, float, float]:
    return (x, 100.0, 12.0, 12.0, conf)


def test_rf7_short_gaps_are_interpolated_and_long_ones_split() -> None:
    from tools.ball_dataset import labels_for

    dets = [_d(0), _d(10), None, None, _d(40), _d(50), _d(60)] + [None] * 6 + [_d(200)] * 2
    labs = labels_for(dets)
    assert sorted(labs) == list(range(7))  # la trayectoria de 5 detecciones + 2 interpolados
    assert labs[2][0] == 20.0 and labs[3][0] == 30.0
    assert 13 not in labs  # 2 detecciones sueltas después de un hueco largo: no alcanzan


def test_rf7_trajectory_without_a_confident_detection_is_rejected() -> None:
    from tools.ball_dataset import labels_for

    assert labels_for([_d(10 * i, conf=0.15) for i in range(8)]) == {}
