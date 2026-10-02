"""Lectura de tramos de video (índices absolutos) y errores de la CLI (SPEC-002 RF-7, manejo de errores)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from volley_cv.__main__ import main
from volley_cv.config import ConfigError
from volley_cv.video.source import probe, read_frames


def make_video(path: Path, n: int = 30, fps: float = 10.0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n):
        frame = np.full((48, 64, 3), i * 8 % 256, dtype=np.uint8)
        w.write(frame)
    w.release()
    return path


def test_read_frames_uses_absolute_indices(tmp_path: Path) -> None:
    video = make_video(tmp_path / "v.avi")
    info = probe(video)
    assert info.fps == pytest.approx(10.0) and (info.width, info.height) == (64, 48)
    got = list(read_frames(video, start_s=1.0, end_s=2.0))
    assert [i for i, _ in got] == list(range(10, 20))
    assert got[0][1].dtype == np.uint8


def test_read_frames_stops_at_end_of_video(tmp_path: Path) -> None:
    video = make_video(tmp_path / "v.avi", n=15)
    assert [i for i, _ in read_frames(video, 1.0, 5.0)] == list(range(10, 15))


def test_probe_missing_video_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No se pudo abrir"):
        probe(tmp_path / "nope.mp4")


def test_cli_without_data_dir_exits_with_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOLLEY_DATA_DIR", str(tmp_path / "no-existe"))
    assert main(["run", "--clip", "A2"]) == 2
    assert "no existe" in capsys.readouterr().err


def test_cli_unknown_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOLLEY_DATA_DIR", str(tmp_path))
    assert main(["run", "--clip", "ZZ9"]) == 2
    assert "clip desconocido" in capsys.readouterr().err


def test_cli_video_requires_config_and_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOLLEY_DATA_DIR", str(tmp_path))
    assert main(["run", "--video", "videos/x.mp4"]) == 2
    assert "--video-config" in capsys.readouterr().err


def test_video_config_missing_and_repo_configs_valid() -> None:
    from volley_cv.video_config import load_video_config

    with pytest.raises(ConfigError, match="Falta la configuración"):
        load_video_config("no_existe")
    for vid in ("jpn_arg_2026", "jpn_kor_2026"):
        cfg = load_video_config(vid)
        assert set(cfg.teams) == {"A", "B"}
