"""Configuración visual por video (configs/videos/<video>.yaml): cancha, exclusiones y colores de equipo."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from volley_cv.config import PROJECT_ROOT, ConfigError

Lab = tuple[float, float, float]


class HsvRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


class CourtConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hsv_ranges: list[HsvRange] = Field(min_length=1)
    margin_px: int = Field(default=20, ge=0)


class TeamColors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    main: Lab
    libero: Lab | None = None


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    court: CourtConfig
    exclude_regions: list[tuple[float, float, float, float]] = Field(default_factory=list)
    teams: dict[str, TeamColors]
    officials: list[Lab] = Field(default_factory=list)  # árbitros / jueces de línea (SPEC-002 RF-3b)
    play_margin: float = Field(default=0.12, ge=0.0, le=0.5)  # zona de juego: fracción del alto (RF-2b)


def load_video_config(video_id: str, root: Path = PROJECT_ROOT) -> VideoConfig:
    path = root / "configs" / "videos" / f"{video_id}.yaml"
    if not path.is_file():
        raise ConfigError(f"Falta la configuración visual del video '{video_id}': se esperaba {path}")
    try:
        cfg = VideoConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (ValidationError, yaml.YAMLError) as e:
        raise ConfigError(f"{path}: configuración de video inválida.\n{e}") from e
    if set(cfg.teams) != {"A", "B"}:
        raise ConfigError(f"{path}: 'teams' debe definir exactamente A y B")
    return cfg
