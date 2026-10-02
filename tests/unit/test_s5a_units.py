"""SPEC-003 AC-4 (reescritura de fusiones), AC-6 (líbero por número), AC-8 (lector easyocr con OCR falso)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from volley_cv.identity.types import JerseyRead, Team
from volley_cv.jersey_reader import EasyOcrJerseyReader
from volley_cv.merges import apply_merges, resolve_chains
from volley_cv.team import TeamClassifier
from volley_cv.video_config import TeamColors

from .test_team import DARK, RED, WHITE, draw

# ---------- AC-4 ----------


def _player(pid: str, x: float) -> dict[str, object]:
    return {
        "player_id": pid,
        "track_id": "track_1",
        "team_id": pid[:6],
        "jersey_number": None,
        "jersey_confidence": 0.0,
        "bbox": [x, 0.0, x + 10, 20.0],
        "confidence": 0.9,
        "state": "TRACKED",
    }


def test_ac4_merges_rewrite_all_frames_and_chains() -> None:
    assert resolve_chains(
        [("TEAM_A_PLAYER_09", "TEAM_A_PLAYER_04"), ("TEAM_A_PLAYER_04", "TEAM_A_PLAYER_01")]
    ) == {
        "TEAM_A_PLAYER_09": "TEAM_A_PLAYER_01",
        "TEAM_A_PLAYER_04": "TEAM_A_PLAYER_01",
    }


def test_ac4_apply_merges_to_jsonl(tmp_path: Path) -> None:
    rows = [
        {"frame": 0, "players": [_player("TEAM_A_PLAYER_01", 0)], "ball": None},
        {
            "frame": 1,
            "players": [_player("TEAM_A_PLAYER_02", 5), _player("TEAM_B_PLAYER_01", 50)],
            "ball": None,
        },
        {"frame": 2, "players": [_player("TEAM_A_PLAYER_01", 6)], "ball": None},
    ]
    p = tmp_path / "frames.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    changed = apply_merges(p, [("TEAM_A_PLAYER_02", "TEAM_A_PLAYER_01")])
    out = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    assert changed == 1
    assert [pl["player_id"] for r in out for pl in r["players"]] == [
        "TEAM_A_PLAYER_01",
        "TEAM_A_PLAYER_01",
        "TEAM_B_PLAYER_01",
        "TEAM_A_PLAYER_01",
    ]
    for r in out:
        ids = [pl["player_id"] for pl in r["players"]]
        assert len(ids) == len(set(ids))


# ---------- AC-6 ----------

TEAMS_NUM = {
    "A": TeamColors(name="Japón", main=WHITE, libero=RED, libero_numbers=[37]),
    "B": TeamColors(name="Argentina", main=DARK, libero=WHITE, libero_numbers=[19]),
}


def test_ac6_libero_number_decides_ambiguous_team() -> None:
    frame, boxes = draw([(WHITE, 300, 600), (WHITE, 700, 600)])
    teams = TeamClassifier(TEAMS_NUM).classify(frame, boxes, numbers=[19, None])
    assert teams == [Team.B, Team.A]  # con #19 es el líbero argentino; sin número, la prior (A)


def test_ac6_libero_number_does_not_override_unambiguous_color() -> None:
    frame, boxes = draw([(DARK, 300, 600)])
    assert TeamClassifier(TEAMS_NUM).classify(frame, boxes, numbers=[37]) == [Team.B]


# Como jpn_kor_2026: Japón rojo; el líbero coreano (#5) azul marino. Japón también tiene un #5 (K2 real)
RED_K, NAVY = (72, 164, 131), (50, 135, 114)
TEAMS_K = {
    "A": TeamColors(name="Japón", main=RED_K, libero=WHITE),
    "B": TeamColors(name="Corea", main=WHITE, libero=NAVY, libero_numbers=[5]),
}


def test_ac6_libero_number_needs_color_at_least_as_close_to_the_libero() -> None:
    # rojo en sombra: dentro del margen de ambigüedad con el azul marino, pero más cerca del rojo
    shadow_red = (63, 152, 124)
    frame, boxes = draw([(shadow_red, 300, 600)])
    assert TeamClassifier(TEAMS_K).classify(frame, boxes, numbers=[5]) == [Team.A]
    frame, boxes = draw([(NAVY, 300, 600)])
    assert TeamClassifier(TEAMS_K).classify(frame, boxes, numbers=[5]) == [Team.B]


# ---------- AC-8 ----------


class FakeOcr:
    def __init__(self, results: list[tuple[object, str, float]]) -> None:
        self.results = results
        self.calls = 0

    def readtext(self, img: np.ndarray, **kw: object) -> list[tuple[object, str, float]]:
        self.calls += 1
        return self.results


FRAME = np.full((400, 400, 3), 120, dtype=np.uint8)
BOX = (100.0, 50.0, 180.0, 300.0)


def test_ac8_emits_confident_one_or_two_digit_reads() -> None:
    r = EasyOcrJerseyReader(min_conf=0.95, ocr=FakeOcr([(None, "7", 0.97), (None, "12", 0.96)])).read(
        FRAME, BOX
    )
    assert r is not None and r.number == 7 and r.confidence == 0.97


def test_ac8_rejects_low_confidence_long_or_non_numeric() -> None:
    for res in ([(None, "7", 0.90)], [(None, "123", 0.99)], [(None, "A", 0.99)], []):
        assert EasyOcrJerseyReader(min_conf=0.95, ocr=FakeOcr(res)).read(FRAME, BOX) is None


def test_ac8_empty_crop_is_not_sent_to_ocr() -> None:
    ocr = FakeOcr([(None, "7", 0.99)])
    assert EasyOcrJerseyReader(ocr=ocr).read(FRAME, (500.0, 500.0, 520.0, 560.0)) is None
    assert ocr.calls == 0


def test_ac8_easyocr_is_loaded_lazily_once(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[object] = []

    class FakeEasyOcr:
        @staticmethod
        def Reader(langs: list[str], gpu: bool, verbose: bool) -> FakeOcr:
            created.append(langs)
            return FakeOcr([(None, "9", 0.99)])

    monkeypatch.setitem(sys.modules, "easyocr", FakeEasyOcr)
    reader = EasyOcrJerseyReader()
    assert created == []  # construir el lector no carga el modelo
    assert [reader.read(FRAME, BOX) for _ in range(2)] == [JerseyRead(9, 0.99)] * 2
    assert created == [["en"]]


def test_ac8_ocr_failure_yields_no_read() -> None:
    class Broken:
        def readtext(self, img: np.ndarray, **kw: object) -> list[tuple[object, str, float]]:
            raise RuntimeError("cuda")

    assert EasyOcrJerseyReader(ocr=Broken()).read(FRAME, BOX) is None


def test_ac4_no_merges_leaves_output_untouched(tmp_path: Path) -> None:
    jsonl = tmp_path / "frames.jsonl"
    jsonl.write_text('{"players": []}\n', encoding="utf-8")
    assert apply_merges(jsonl, []) == 0
    assert jsonl.read_text(encoding="utf-8") == '{"players": []}\n'


def test_votes_removal_of_unknown_number_is_ignored() -> None:
    from volley_cv.identity.jersey import JerseyVotes

    votes = JerseyVotes()
    votes.add(JerseyRead(7, 0.9))
    votes.remove(JerseyRead(11, 0.9))  # nunca sumado: no debe quedar en negativo
    votes.remove(JerseyRead(7, 0.9))
    assert votes.counts == {} and votes.conf_sum == {}
