from pathlib import Path

import pytest

from volley_cv.config import ConfigError, load_clips, load_settings, parse_timestamp


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadSettings:
    def test_env_var_takes_precedence_over_local_yaml(self, tmp_path: Path) -> None:
        env_dir = tmp_path / "from_env"
        env_dir.mkdir()
        yaml_dir = tmp_path / "from_yaml"
        yaml_dir.mkdir()
        write(tmp_path / "configs" / "local.yaml", f"data_dir: {yaml_dir.as_posix()}\n")

        s = load_settings(project_root=tmp_path, env={"VOLLEY_DATA_DIR": str(env_dir)})

        assert s.data_dir == env_dir

    def test_falls_back_to_local_yaml(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        write(tmp_path / "configs" / "local.yaml", f"data_dir: {data.as_posix()}\n")

        s = load_settings(project_root=tmp_path, env={})

        assert s.data_dir == data
        assert s.videos_dir == data / "videos"

    def test_missing_configuration_raises_actionable_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="VOLLEY_DATA_DIR"):
            load_settings(project_root=tmp_path, env={})

    def test_nonexistent_data_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="no existe"):
            load_settings(project_root=tmp_path, env={"VOLLEY_DATA_DIR": str(tmp_path / "nope")})


class TestParseTimestamp:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0:00", 0.0), ("9:05", 545.0), ("35:18", 2118.0), ("1:02:03", 3723.0), (12.5, 12.5), (7, 7.0)],
    )
    def test_valid(self, raw: str | float, expected: float) -> None:
        assert parse_timestamp(raw) == expected

    @pytest.mark.parametrize("raw", ["", "abc", "1:75", "-0:05", -3])
    def test_invalid(self, raw: str | float) -> None:
        with pytest.raises(ValueError):
            parse_timestamp(raw)


CLIPS_OK = """
videos:
  jpn_arg_2026:
    file: videos/external/jpn_arg_2026.mp4
clips:
  - id: A1
    video: jpn_arg_2026
    start: "15:21"
    duration_s: 20
    scenarios: [tracking_normal, numbers_visible]
    verified: true
"""


class TestLoadClips:
    def test_loads_and_resolves_against_data_dir(self, tmp_path: Path) -> None:
        cfg = write(tmp_path / "clips.yaml", CLIPS_OK)

        catalog = load_clips(cfg)

        clip = catalog.clips[0]
        assert clip.id == "A1"
        assert clip.start_s == 921.0
        assert clip.end_s == 941.0
        assert catalog.video_path(clip, data_dir=tmp_path) == tmp_path / "videos/external/jpn_arg_2026.mp4"

    def test_unknown_video_reference_is_rejected(self, tmp_path: Path) -> None:
        cfg = write(tmp_path / "clips.yaml", CLIPS_OK.replace("video: jpn_arg_2026", "video: missing"))
        with pytest.raises(ConfigError, match="missing"):
            load_clips(cfg)

    def test_unknown_scenario_is_rejected(self, tmp_path: Path) -> None:
        cfg = write(tmp_path / "clips.yaml", CLIPS_OK.replace("numbers_visible", "made_up"))
        with pytest.raises(ConfigError):
            load_clips(cfg)

    def test_duplicate_clip_ids_are_rejected(self, tmp_path: Path) -> None:
        dup = CLIPS_OK + CLIPS_OK.split("clips:\n", 1)[1]
        cfg = write(tmp_path / "clips.yaml", dup)
        with pytest.raises(ConfigError, match="duplicado"):
            load_clips(cfg)

    def test_absolute_video_paths_are_rejected(self, tmp_path: Path) -> None:
        cfg = write(tmp_path / "clips.yaml", CLIPS_OK.replace("videos/external", "D:/abs/videos"))
        with pytest.raises(ConfigError, match="relativa"):
            load_clips(cfg)


def test_repo_clip_catalog_is_valid() -> None:
    """El catálogo versionado debe ser siempre válido (regresión de configuración)."""
    root = Path(__file__).resolve().parents[2]
    catalog = load_clips(root / "configs" / "eval" / "clips.yaml")
    assert len(catalog.clips) == 10
