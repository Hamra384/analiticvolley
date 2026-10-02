"""AC-15 (SPEC-001): la salida por frame valida contra el esquema del punto 13 del MVP."""

import json

import pytest
from pydantic import ValidationError

from volley_cv.output.schema import FrameOutput, PlayerOut

VALID = {
    "frame": 1250,
    "players": [
        {
            "player_id": "TEAM_A_PLAYER_04",
            "track_id": "track_17",
            "team_id": "TEAM_A",
            "jersey_number": 7,
            "jersey_confidence": 0.94,
            "bbox": [10, 20, 60, 170],
            "confidence": 0.91,
            "state": "TRACKED",
        },
        {
            "player_id": "TEAM_B_PLAYER_03",
            "track_id": "track_21.1",
            "team_id": "TEAM_B",
            "jersey_number": None,
            "jersey_confidence": 0,
            "bbox": [100, 20, 160, 170],
            "confidence": 0.8,
            "state": "REIDENTIFIED",
        },
    ],
    "ball": {"track_id": "ball_01", "position": [640.0, 300.0], "confidence": 0.88, "state": "PREDICTED"},
}


def test_mvp_example_validates_and_roundtrips_json() -> None:
    fo = FrameOutput.model_validate(VALID)
    again = FrameOutput.model_validate(json.loads(fo.model_dump_json()))
    assert again == fo
    assert fo.players[1].jersey_number is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("player_id", "PLAYER_7"),  # no sigue TEAM_X_PLAYER_NN
        ("player_id", "TEAM_C_PLAYER_01"),
        ("track_id", "17"),
        ("team_id", "TEAM_C"),
        ("state", "VISIBLE"),
        ("jersey_confidence", 1.5),
        ("jersey_number", 120),
        ("confidence", -0.1),
    ],
)
def test_invalid_player_fields_are_rejected(field: str, value: object) -> None:
    data = dict(VALID["players"][0])  # type: ignore[index]
    data[field] = value
    with pytest.raises(ValidationError):
        PlayerOut.model_validate(data)


def test_ball_predicted_state_is_distinct_from_detected() -> None:
    data = json.loads(json.dumps(VALID))
    data["ball"]["state"] = "SEEN"
    with pytest.raises(ValidationError):
        FrameOutput.model_validate(data)


def test_extra_fields_are_rejected() -> None:
    data = json.loads(json.dumps(VALID))
    data["players"][0]["name"] = "Ishikawa"
    with pytest.raises(ValidationError):
        FrameOutput.model_validate(data)
