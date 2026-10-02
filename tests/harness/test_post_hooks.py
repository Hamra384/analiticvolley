import json
import shutil
from pathlib import Path

import pytest

from .conftest import ROOT, run_hook

needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="requiere uv en PATH (instalado en CI, #2)")


@needs_uv
def test_lint_blocks_with_unfixable_error(tmp_path: Path) -> None:
    f = ROOT / "src" / "volley_cv" / "_tmp_lint_probe.py"
    f.write_text("def f() -> int:\n    return undefined_name\n", encoding="utf-8")
    try:
        code, out, _ = run_hook(
            "post_edit_lint.py", {"tool_name": "Write", "tool_input": {"file_path": str(f)}}, cwd=ROOT
        )
    finally:
        f.unlink()
    assert code == 0 and out is not None
    assert out["decision"] == "block"
    assert "F821" in out["reason"]


@needs_uv
def test_lint_autofixes_and_stays_silent(tmp_path: Path) -> None:
    f = ROOT / "src" / "volley_cv" / "_tmp_lint_probe2.py"
    f.write_text("import os\nx=1\n", encoding="utf-8")
    try:
        code, out, _ = run_hook(
            "post_edit_lint.py", {"tool_name": "Edit", "tool_input": {"file_path": str(f)}}, cwd=ROOT
        )
        fixed = f.read_text(encoding="utf-8")
    finally:
        f.unlink()
    assert code == 0 and out is None
    assert fixed == "x = 1\n"


@pytest.mark.parametrize("rel", ["README.md", "legacy/run_deteccion.py"])
def test_lint_ignores_non_python_and_legacy(rel: str) -> None:
    code, out, _ = run_hook(
        "post_edit_lint.py", {"tool_name": "Edit", "tool_input": {"file_path": str(ROOT / rel)}}, cwd=ROOT
    )
    assert code == 0 and out is None


def test_log_event_records_metadata_only(tmp_path: Path) -> None:
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "s1",
        "tool_name": "Bash",
        "tool_input": {"command": "echo SECRET_TOKEN=abc"},
        "tool_response": {"stdout": "SECRET_TOKEN=abc", "stderr": "", "interrupted": False},
    }
    code, out, _ = run_hook("log_event.py", payload, cwd=tmp_path)
    assert code == 0 and out is None
    lines = (tmp_path / ".harness-test-logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[-1])
    assert rec["event"] == "PostToolUse" and rec["tool"] == "Bash" and rec["session"] == "s1"
    assert "SECRET_TOKEN" not in lines[-1]


def test_log_event_never_fails_on_garbage(tmp_path: Path) -> None:
    code, out, _ = run_hook("log_event.py", {"unexpected": [1, 2, 3]}, cwd=tmp_path)
    assert code == 0 and out is None
