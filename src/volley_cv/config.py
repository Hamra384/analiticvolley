"""Configuración: ubicación de datos y catálogo de clips de evaluación.

Los videos y datos viven fuera del repo. Su ubicación se resuelve, en orden:
1. variable de entorno ``VOLLEY_DATA_DIR``;
2. ``configs/local.yaml`` (gitignored) con la clave ``data_dir``.
Nunca se usan rutas absolutas hardcodeadas en el código ni en configs versionadas.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ENV_DATA_DIR = "VOLLEY_DATA_DIR"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    """Configuración ausente o inválida, con un mensaje accionable."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: Path

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"


def load_settings(project_root: Path = PROJECT_ROOT, env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    raw = env.get(ENV_DATA_DIR)
    source = f"la variable de entorno {ENV_DATA_DIR}"
    if not raw:
        local = project_root / "configs" / "local.yaml"
        if local.is_file():
            data = _read_yaml(local)
            raw = data.get("data_dir") if isinstance(data, dict) else None
            source = str(local)
    if not raw:
        raise ConfigError(
            f"No se encontró el directorio de datos. Definí {ENV_DATA_DIR} o creá "
            f"configs/local.yaml con 'data_dir: <ruta>'."
        )
    data_dir = Path(str(raw))
    if not data_dir.is_dir():
        raise ConfigError(f"El directorio de datos indicado en {source} no existe: {data_dir}")
    return Settings(data_dir=data_dir)


_TS = re.compile(r"^(?:(\d+):)?(\d+):(\d{2})(?:\.(\d+))?$")


def parse_timestamp(value: str | float) -> float:
    """Convierte 'mm:ss', 'h:mm:ss' o segundos a segundos (float)."""
    if isinstance(value, int | float):
        if value < 0:
            raise ValueError(f"timestamp negativo: {value}")
        return float(value)
    m = _TS.match(value.strip())
    if not m:
        raise ValueError(f"timestamp inválido: {value!r} (usar mm:ss o h:mm:ss)")
    hours, minutes, seconds, frac = m.groups()
    if int(seconds) >= 60 or (hours is not None and int(minutes) >= 60):
        raise ValueError(f"timestamp inválido: {value!r}")
    total = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    return total + (float(f"0.{frac}") if frac else 0.0)


class Scenario(StrEnum):
    TRACKING_NORMAL = "tracking_normal"
    OCCLUSION = "occlusion"
    OVERLAP = "overlap"
    NUMBERS_VISIBLE = "numbers_visible"
    NUMBERS_INVISIBLE = "numbers_invisible"
    NUMBER_APPEARS_LATER = "number_appears_later"
    DETECTOR_DROPOUT = "detector_dropout"
    BALL_PARTIAL_LOSS = "ball_partial_loss"
    BALL_FAST = "ball_fast"
    BALL_NEAR_NET = "ball_near_net"
    NEW_PLAYER = "new_player"
    SUBSTITUTION = "substitution"
    LIBERO = "libero"
    CAMERA_PAN = "camera_pan"
    CELEBRATION_CLUSTER = "celebration_cluster"
    OUT_OF_FRAME = "out_of_frame"


class VideoRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    source: str | None = None
    resolution: str | None = None
    notes: str | None = None

    @field_validator("file")
    @classmethod
    def _relative(cls, v: str) -> str:
        if PureWindowsPath(v).is_absolute() or PurePosixPath(v).is_absolute() or PureWindowsPath(v).drive:
            raise ValueError(f"la ruta del video debe ser relativa al directorio de datos: {v}")
        return v


class Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    video: str
    start: str | float
    duration_s: float = Field(gt=0)
    scenarios: list[Scenario] = Field(min_length=1)
    verified: bool = False
    notes: str | None = None

    @property
    def start_s(self) -> float:
        return parse_timestamp(self.start)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @field_validator("start")
    @classmethod
    def _valid_start(cls, v: str | float) -> str | float:
        parse_timestamp(v)
        return v


class ClipCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    videos: dict[str, VideoRef]
    clips: list[Clip]

    def video_path(self, clip: Clip, data_dir: Path) -> Path:
        return data_dir / self.videos[clip.video].file


def load_clips(path: Path) -> ClipCatalog:
    data = _read_yaml(path)
    try:
        catalog = ClipCatalog.model_validate(data)
    except ValidationError as e:
        msg = str(e)
        if "relativa" in msg:
            raise ConfigError(
                f"{path}: la ruta del video debe ser relativa al directorio de datos.\n{e}"
            ) from e
        raise ConfigError(f"{path}: catálogo de clips inválido.\n{e}") from e
    seen: set[str] = set()
    for clip in catalog.clips:
        if clip.id in seen:
            raise ConfigError(f"{path}: id de clip duplicado: {clip.id}")
        seen.add(clip.id)
        if clip.video not in catalog.videos:
            raise ConfigError(f"{path}: el clip {clip.id} referencia un video inexistente: {clip.video}")
    return catalog


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"No se pudo leer {path}: {e}") from e
