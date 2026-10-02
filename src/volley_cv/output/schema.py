"""Esquema de salida por frame (punto 13 del MVP; SPEC-001 AC-15)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PlayerStateName = Literal["DETECTED", "TRACKED", "OCCLUDED", "LOST", "REIDENTIFIED"]
BallStateName = Literal["DETECTED", "TRACKED", "PREDICTED", "LOST", "REACQUIRED"]


class PlayerOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    player_id: str = Field(pattern=r"^TEAM_[AB]_PLAYER_\d{2,}$")
    track_id: str = Field(pattern=r"^track_\d+(\.\d+)?$")
    team_id: Literal["TEAM_A", "TEAM_B"]
    jersey_number: int | None = Field(default=None, ge=0, le=99)
    jersey_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    state: PlayerStateName


class BallOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: str
    position: tuple[float, float] | None
    confidence: float = Field(ge=0.0, le=1.0)
    state: BallStateName


class FrameOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame: int = Field(ge=0)
    players: list[PlayerOut]
    ball: BallOut | None = None
