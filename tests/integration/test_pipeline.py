"""AC-5 y AC-6 (SPEC-002): pipeline completo con detector/tracker/embedder falsos sobre frames sintéticos."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from volley_cv.court import CourtMask
from volley_cv.identity import IdentityManager
from volley_cv.output.schema import FrameOutput
from volley_cv.pipeline import Pipeline
from volley_cv.team import TeamClassifier
from volley_cv.video_config import CourtConfig, HsvRange, TeamColors

W, H, N = 640, 360, 60
CUT_AT = 40
WHITE, DARK = (190, 130, 128), (60, 130, 124)
TEAMS = {"A": TeamColors(name="A", main=WHITE), "B": TeamColors(name="B", main=DARK)}
COURT = CourtConfig(hsv_ranges=[HsvRange(lower=(3, 25, 80), upper=(18, 255, 255))], margin_px=5)
# (x, y_pies, color, embedding)
PLAYERS = [
    (150.0, 300.0, WHITE, 0),
    (400.0, 310.0, WHITE, 1),
    (250.0, 160.0, DARK, 2),
    (450.0, 170.0, DARK, 3),
]
BENCH = (20.0, 300.0, WHITE, 4)  # fuera de la cancha: no debe convertirse en jugador


def lab_to_bgr(lab: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[lab]]), cv2.COLOR_LAB2BGR)[0, 0])  # type: ignore[return-value]


def make_frame(i: int) -> np.ndarray:
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    # otra toma desde CUT_AT: cambia bruscamente el fondo (público/carteles), no el color de la cancha
    hsv[:] = (85, 120, 120) if i < CUT_AT else (140, 200, 30)  # el corte cambia el brillo (no isoluminante)
    hsv[100:340, 100:560] = (10, 150, 200)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for x, feet, lab, _ in [*PLAYERS, BENCH]:
        dx = i * 0.5
        cv2.rectangle(
            frame, (int(x + dx), int(feet - 80)), (int(x + dx + 30), int(feet)), lab_to_bgr(lab), -1
        )
    return frame


def boxes(i: int) -> list[tuple[float, float, float, float]]:
    return [(x + i * 0.5, feet - 80, x + i * 0.5 + 30, feet) for x, feet, _, _ in [*PLAYERS, BENCH]]


def who(box: Sequence[float]) -> int:
    """Índice del jugador de referencia (0-3) o del banco (4) más cercano a la caja.

    Los jugadores se desplazan como mucho 30 px en x durante la secuencia y están separados >= 100 px.
    """
    refs = [*PLAYERS, BENCH]
    return min(range(len(refs)), key=lambda j: abs(refs[j][0] - box[0]) + abs(refs[j][1] - box[3]))


class FakeDetector:
    def __init__(self) -> None:
        self.i = 0

    def detect(self, frame: np.ndarray) -> np.ndarray:
        dets = np.array([[*b, 0.9] for b in boxes(self.i)], dtype=np.float32)
        self.i += 1
        return dets


class FakeTracker:
    """Detecciones con ID estable por persona; tras un reset, IDs nuevos (como un tracker real)."""

    def __init__(self) -> None:
        self.offset = 0

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        rows = [[*d[:4], self.offset + who(d) + 1, d[4]] for d in detections]
        return np.array(rows, dtype=np.float32).reshape(-1, 6)

    def reset(self) -> None:
        self.offset += 100


class FakeEmbedder:
    """Un vector one-hot por persona: apariencia perfecta (el foco de este test es el cableado)."""

    def embed(
        self, frame: np.ndarray, bxs: Sequence[tuple[float, float, float, float]]
    ) -> list[np.ndarray | None]:
        out: list[np.ndarray | None] = []
        for b in bxs:
            v = np.zeros(8, dtype=np.float32)
            v[who(b)] = 1.0
            out.append(v)
        return out


class SpyManager(IdentityManager):
    def __init__(self) -> None:
        super().__init__()
        self.resets_at: list[int] = []
        self._current = -1

    def update(self, frame, observations):  # type: ignore[no-untyped-def]
        self._current = frame
        return super().update(frame, observations)

    def reset(self) -> None:
        self.resets_at.append(self._current + 1)
        super().reset()


def run(tmp_path: Path) -> tuple[list[FrameOutput], SpyManager, object]:
    mgr = SpyManager()
    pipe = Pipeline(
        detector=FakeDetector(),
        tracker=FakeTracker(),
        embedder=FakeEmbedder(),
        court=CourtMask(COURT),
        teams=TeamClassifier(TEAMS),
        manager=mgr,
    )
    summary = pipe.run(((1000 + i, make_frame(i)) for i in range(N)), tmp_path, fps=30.0, write_video=False)
    lines = (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    return [FrameOutput.model_validate(json.loads(line)) for line in lines], mgr, summary


def test_ac5_one_valid_line_per_frame_with_absolute_frame_index(tmp_path: Path) -> None:
    outs, _, summary = run(tmp_path)
    assert len(outs) == N
    assert [o.frame for o in outs] == list(range(1000, 1000 + N))
    assert summary.frames == N  # type: ignore[attr-defined]


def test_ac5_four_players_with_stable_ids_and_bench_excluded(tmp_path: Path) -> None:
    outs, _, _ = run(tmp_path)
    before_cut = outs[10:CUT_AT]
    ids = [{p.player_id for p in o.players} for o in before_cut]
    assert all(len(s) == 4 for s in ids)
    assert len(set.union(*ids)) == 4
    teams = {p.player_id[:6] for o in before_cut for p in o.players}
    assert teams == {"TEAM_A", "TEAM_B"}


def test_debug_video_is_written_with_all_frames(tmp_path: Path) -> None:
    pipe = Pipeline(
        detector=FakeDetector(),
        tracker=FakeTracker(),
        embedder=FakeEmbedder(),
        court=CourtMask(COURT),
        teams=TeamClassifier(TEAMS),
    )
    summary = pipe.run(((i, make_frame(i)) for i in range(N)), tmp_path, fps=30.0, write_video=True)
    assert summary.video == tmp_path / "debug.mp4"
    cap = cv2.VideoCapture(str(summary.video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (W, H)
    cap.release()


def test_cli_end_to_end_with_fake_models(tmp_path: Path, monkeypatch: object) -> None:
    """CLI completa (`--video`) con los adaptadores de modelos reemplazados por los falsos."""
    import volley_cv.adapters as adapters
    from volley_cv.__main__ import main

    (tmp_path / "videos").mkdir()
    w = cv2.VideoWriter(str(tmp_path / "videos" / "s.avi"), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (W, H))
    for i in range(N):
        w.write(make_frame(i))
    w.release()
    mp = monkeypatch  # pytest.MonkeyPatch
    mp.setenv("VOLLEY_DATA_DIR", str(tmp_path))  # type: ignore[attr-defined]
    mp.setattr(adapters, "YoloPersonDetector", lambda weights, **kw: FakeDetector())  # type: ignore[attr-defined]
    mp.setattr(adapters, "ByteTrackTracker", lambda frame_rate=30: FakeTracker())  # type: ignore[attr-defined]

    class FakeBall:
        def detect(self, frame: np.ndarray) -> np.ndarray:
            return np.array([[320.0, 100.0, 10.0, 10.0, 0.8]], dtype=np.float32)

    # SPEC-005: la pelota sale de la misma pasada del detector de personas
    mp.setattr(adapters.YoloBallDetector, "from_person_detector", classmethod(lambda cls, p: FakeBall()))  # type: ignore[attr-defined]
    import volley_cv.appearance as appearance

    mp.setattr(appearance, "ColorHistEmbedder", FakeEmbedder)  # type: ignore[attr-defined]
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "--video",
            "videos/s.avi",
            "--video-config",
            "jpn_arg_2026",
            "--end",
            "0:01",
            "--out",
            str(out),
            "--ball-coco",
        ]
    )
    assert code == 0
    lines = (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    assert (out / "debug.mp4").is_file()
    assert json.loads(lines[-1])["ball"]["state"] == "TRACKED"


def test_ac6_reset_called_at_cut_and_identities_recovered(tmp_path: Path) -> None:
    outs, mgr, summary = run(tmp_path)
    assert summary.cuts == [1000 + CUT_AT]  # type: ignore[attr-defined]
    assert mgr.resets_at == [1000 + CUT_AT]
    before = {p.player_id for p in outs[CUT_AT - 1].players}
    after = {p.player_id for p in outs[N - 1].players}
    assert before == after  # misma apariencia tras el corte -> mismas identidades (Re-ID por apariencia)
