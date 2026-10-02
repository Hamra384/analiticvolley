"""Regresión: .gitignore no debe ocultar documentación (bug S2: 'data/' ignoraba docs/data/)."""

import subprocess

import pytest

from .conftest import ROOT


def ignored(path: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True)
    return r.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "docs/data/DATASETS.md",
        "docs/data/ANNOTATION.md",
        "docs/spikes/001-tracker.md",
        "src/volley_cv/data/x.py",
    ],
)
def test_documentation_and_code_are_not_ignored(path: str) -> None:
    assert not ignored(path), path


@pytest.mark.parametrize(
    "path",
    [
        "data/frames/1.jpg",
        "videos/partido.mp4",
        "clip.mkv",
        "docs/x.mp4",
        "training_data/7/1.jpg",
        "configs/local.yaml",
    ],
)
def test_media_and_local_data_are_ignored(path: str) -> None:
    assert ignored(path), path
