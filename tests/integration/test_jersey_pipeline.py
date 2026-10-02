"""SPEC-003 AC-7 (cuándo se llama al lector) y reescritura con fusiones al final del pipeline (AC-4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from volley_cv.court import CourtMask
from volley_cv.identity import IdentityManager
from volley_cv.identity.types import JerseyRead
from volley_cv.pipeline import Pipeline
from volley_cv.team import TeamClassifier

from .test_pipeline import COURT, CUT_AT, TEAMS, FakeDetector, FakeEmbedder, FakeTracker, N, make_frame


class CountingReader:
    def __init__(self, number: int | None = 7) -> None:
        self.calls: list[tuple[int, int]] = []  # (alto de la caja, frame)
        self.number = number
        self.frame = -1

    def read(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> JerseyRead | None:
        self.calls.append((round(box[3] - box[1]), self.frame))
        return JerseyRead(self.number, 0.97) if self.number is not None else None


def run(tmp_path: Path, reader: CountingReader, min_height: float, stride: int) -> Pipeline:
    pipe = Pipeline(
        FakeDetector(),
        FakeTracker(),
        FakeEmbedder(),
        CourtMask(COURT),
        TeamClassifier(TEAMS),
        jersey_reader=reader,
        read_min_height=min_height,
        read_stride=stride,
    )
    pipe.run(((i, make_frame(i)) for i in range(N)), tmp_path, write_video=False)
    return pipe


def test_ac7_small_boxes_are_not_read(tmp_path: Path) -> None:
    reader = CountingReader()
    run(tmp_path, reader, min_height=0.5, stride=1)  # cajas de 80 px en frames de 360: 22 % < 50 %
    assert reader.calls == []


def test_ac7_each_track_is_read_at_most_once_every_stride_frames(tmp_path: Path) -> None:
    reader = CountingReader()
    run(tmp_path, reader, min_height=0.1, stride=5)
    # 4 jugadores en cancha (el banco no llega al tracker) x N frames / stride, con margen por el corte
    assert 0 < len(reader.calls) <= 4 * (N // 5 + 2)


def test_jersey_numbers_reach_the_output(tmp_path: Path) -> None:
    reader = CountingReader(number=7)
    run(tmp_path, reader, min_height=0.1, stride=1)
    rows = [json.loads(x) for x in (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    numbers = {p["jersey_number"] for r in rows[20:35] for p in r["players"]}
    assert 7 in numbers  # al menos una identidad tomó el 7 (las demás quedan en null por unicidad por equipo)


def test_merges_file_is_written(tmp_path: Path) -> None:
    run(tmp_path, CountingReader(number=None), min_height=0.1, stride=1)
    assert json.loads((tmp_path / "merges.json").read_text(encoding="utf-8")) == []


class MergingManager(IdentityManager):
    """Manager real que además informa una fusión, para verificar la reescritura al final del run."""

    @property
    def merges(self) -> list[tuple[str, str, int]]:
        return [("TEAM_A_PLAYER_02", "TEAM_A_PLAYER_01", 10)]


def test_ac4_pipeline_rewrites_output_with_merges(tmp_path: Path) -> None:
    pipe = Pipeline(
        FakeDetector(),
        FakeTracker(),
        FakeEmbedder(),
        CourtMask(COURT),
        TeamClassifier(TEAMS),
        MergingManager(),
    )
    pipe.run(((i, make_frame(i)) for i in range(N)), tmp_path, write_video=False)
    text = (tmp_path / "frames.jsonl").read_text(encoding="utf-8")
    assert "TEAM_A_PLAYER_01" in text and "TEAM_A_PLAYER_02" not in text
    assert json.loads((tmp_path / "merges.json").read_text(encoding="utf-8")) == [
        {"from": "TEAM_A_PLAYER_02", "to": "TEAM_A_PLAYER_01", "frame": 10}
    ]


def _pipe(reader: CountingReader) -> Pipeline:
    return Pipeline(
        FakeDetector(),
        FakeTracker(),
        FakeEmbedder(),
        CourtMask(COURT),
        TeamClassifier(TEAMS),
        jersey_reader=reader,
        read_min_height=0.1,
    )


def test_ac10_torso_covered_by_closer_person_is_not_read() -> None:
    # A2 real: el #35 de un japonés en primer plano tapaba el torso de un argentino del fondo (heredó el 35)
    reader = CountingReader()
    back = (100.0, 50.0, 140.0, 150.0)  # pies en y=150 (más lejos de la cámara)
    front = (110.0, 70.0, 160.0, 200.0)  # pies en y=200 (más cerca): tapa el torso de `back`
    tracks = np.array([[*back, 1, 0.9], [*front, 2, 0.9]], dtype=np.float32)
    reads = _pipe(reader)._read_numbers(np.zeros((360, 640, 3), np.uint8), tracks, [back, front])
    assert reads[0] is None and reads[1] is not None
    assert [h for h, _ in reader.calls] == [130]  # solo se leyó la persona de adelante


def test_ac10_slight_overlap_still_reads() -> None:
    reader = CountingReader()
    a = (100.0, 50.0, 140.0, 150.0)
    b = (136.0, 60.0, 180.0, 200.0)  # roza el borde: tapa < 15 % del torso de `a`
    tracks = np.array([[*a, 1, 0.9], [*b, 2, 0.9]], dtype=np.float32)
    reads = _pipe(reader)._read_numbers(np.zeros((360, 640, 3), np.uint8), tracks, [a, b])
    assert reads[0] is not None and reads[1] is not None


class SpyTeams(TeamClassifier):
    def __init__(self) -> None:
        super().__init__(TEAMS)
        self.seen: list[list[int | None]] = []

    def classify(self, frame, boxes, numbers=None):  # type: ignore[no-untyped-def]
        self.seen.append(list(numbers or []))
        return super().classify(frame, boxes, numbers)


class FirstReadOnly(CountingReader):
    """Lee el número una sola vez por caja (como el OCR real: pocas lecturas confiables)."""

    def read(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> JerseyRead | None:
        out = super().read(frame, box)
        return out if len(self.calls) <= 4 else None


def test_ac11_last_read_number_is_kept_for_the_track_until_a_cut(tmp_path: Path) -> None:
    # A3 real: el líbero #19 se leía en pocos frames y entre lecturas el color ambiguo lo mandaba al rival
    teams = SpyTeams()
    pipe = Pipeline(
        FakeDetector(),
        FakeTracker(),
        FakeEmbedder(),
        CourtMask(COURT),
        teams,
        jersey_reader=FirstReadOnly(),
        read_min_height=0.1,
        read_stride=5,
    )
    pipe.run(((i, make_frame(i)) for i in range(N)), tmp_path, write_video=False)
    assert all(ns[:4] == [7] * 4 for ns in teams.seen[1:CUT_AT])  # recordado entre lecturas (los 4 leídos)
    assert all(7 not in ns for ns in teams.seen[CUT_AT:])  # el corte lo olvida (los tracks son otros)


def test_review_m3_overlap_forgets_the_remembered_number() -> None:
    # en una superposición el tracker puede pasar el ID a otra persona: el número recordado ya no es confiable
    pipe = _pipe(CountingReader())
    a, b = (100.0, 50.0, 140.0, 150.0), (300.0, 50.0, 340.0, 150.0)
    tracks = np.array([[*a, 1, 0.9], [*b, 2, 0.9]], dtype=np.float32)
    assert pipe._team_numbers(tracks, [a, b], [JerseyRead(19, 0.97), None]) == [19, None]
    assert pipe._team_numbers(tracks, [a, b], [None, None]) == [19, None]  # recordado
    near = (105.0, 50.0, 145.0, 150.0)  # se superpone con `a`
    tracks2 = np.array([[*a, 1, 0.9], [*near, 2, 0.9]], dtype=np.float32)
    assert pipe._team_numbers(tracks2, [a, near], [None, None]) == [None, None]
    assert pipe._team_numbers(tracks, [a, b], [None, None]) == [None, None]  # olvidado, no vuelve
