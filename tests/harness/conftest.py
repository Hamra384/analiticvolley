"""Carga los hooks de .claude/hooks como módulos (no son un paquete importable)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".claude" / "hooks"
SCRIPTS = ROOT / "scripts"


def load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def guard_bash() -> ModuleType:
    return load(HOOKS / "guard_bash.py")


@pytest.fixture(scope="session")
def guard_edit() -> ModuleType:
    return load(HOOKS / "guard_edit.py")


def run_hook(
    script: str, payload: dict[str, Any], cwd: Path = ROOT
) -> tuple[int, dict[str, Any] | None, str]:
    """Ejecuta un hook como lo haría Claude Code: JSON por stdin, decisión por stdout."""
    proc = subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
        env={
            **os.environ,
            # los tests que corren desde la raíz del repo no deben dejar artefactos en el árbol
            "HARNESS_LOG_DIR": str(
                Path(tempfile.mkdtemp(prefix="harness-logs-")) if cwd == ROOT else cwd / ".harness-test-logs"
            ),
            "CLAUDE_PROJECT_DIR": str(cwd),
        },
    )
    out = json.loads(proc.stdout) if proc.stdout.strip() else None
    return proc.returncode, out, proc.stderr
