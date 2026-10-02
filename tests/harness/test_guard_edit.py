from pathlib import Path
from types import ModuleType

import pytest

from .conftest import run_hook

ROOT = Path("D:/repo")


def ev(mod: ModuleType, tool: str, rel: str, **kw: str) -> object:
    return mod.evaluate(tool, {"file_path": str(ROOT / rel), **kw}, project_root=ROOT)


@pytest.mark.parametrize(
    "rel",
    [
        ".claude/settings.json",
        ".claude/hooks/guard_bash.py",
        ".github/workflows/ci.yml",
        ".github/CODEOWNERS",
        "scripts/check_tests_ran.py",
        "quality/baseline.json",
    ],
)
def test_protected_harness_files_require_approval(guard_edit: ModuleType, rel: str) -> None:
    d = ev(guard_edit, "Edit", rel, old_string="a", new_string="b")
    assert d is not None and d.decision == "ask"


@pytest.mark.parametrize("rel", ["videos/clip.mp4", "data/x.mov", ".env", ".env.local", "configs/.env"])
def test_media_and_secrets_are_denied(guard_edit: ModuleType, rel: str) -> None:
    d = ev(guard_edit, "Write", rel, content="x")
    assert d is not None and d.decision == "deny"


@pytest.mark.parametrize("rel", ["src/volley_cv/config.py", "docs/adr/0004-x.md", "tests/unit/test_new.py"])
def test_ordinary_files_pass(guard_edit: ModuleType, rel: str) -> None:
    assert ev(guard_edit, "Write", rel, content="def test_a():\n    assert 1\n") is None


class TestWeakeningTests:
    @pytest.mark.parametrize(
        "added",
        [
            "@pytest.mark.skip\n",
            "@pytest.mark.skip(reason='later')\n",
            "    pytest.skip('flaky')\n",
            "@pytest.mark.xfail\n",
            "@unittest.skip('x')\n",
        ],
    )
    def test_adding_skip_or_xfail_requires_approval(self, guard_edit: ModuleType, added: str) -> None:
        d = ev(
            guard_edit,
            "Edit",
            "tests/unit/test_x.py",
            old_string="def test_a():\n",
            new_string=added + "def test_a():\n",
        )
        assert d is not None and d.decision == "ask"

    def test_skip_with_issue_reference_is_allowed(self, guard_edit: ModuleType) -> None:
        d = ev(
            guard_edit,
            "Edit",
            "tests/unit/test_x.py",
            old_string="def test_a():\n",
            new_string="@pytest.mark.skip(reason='bloqueado por #12')\ndef test_a():\n",
        )
        assert d is None

    def test_removing_asserts_requires_approval(self, guard_edit: ModuleType) -> None:
        d = ev(
            guard_edit,
            "Edit",
            "tests/unit/test_x.py",
            old_string="    assert a == 1\n    assert b == 2\n",
            new_string="    assert a == 1\n",
        )
        assert d is not None and d.decision == "ask"

    def test_rewriting_file_with_fewer_tests_requires_approval(
        self, guard_edit: ModuleType, tmp_path: Path
    ) -> None:
        f = tmp_path / "tests" / "unit" / "test_y.py"
        f.parent.mkdir(parents=True)
        f.write_text("def test_a():\n    assert 1\n\ndef test_b():\n    assert 2\n", encoding="utf-8")
        d = guard_edit.evaluate(
            "Write",
            {"file_path": str(f), "content": "def test_a():\n    assert 1\n"},
            project_root=tmp_path,
        )
        assert d is not None and d.decision == "ask"

    def test_adding_tests_is_allowed(self, guard_edit: ModuleType) -> None:
        d = ev(
            guard_edit,
            "Edit",
            "tests/unit/test_x.py",
            old_string="    assert a == 1\n",
            new_string="    assert a == 1\n\n\ndef test_b():\n    assert b == 2\n",
        )
        assert d is None

    def test_non_test_python_is_not_checked_for_asserts(self, guard_edit: ModuleType) -> None:
        d = ev(guard_edit, "Edit", "src/volley_cv/x.py", old_string="assert x\n", new_string="")
        assert d is None


def test_protocol_end_to_end(tmp_path: Path) -> None:
    code, out, _ = run_hook(
        "guard_edit.py",
        {"tool_name": "Write", "tool_input": {"file_path": str(tmp_path / ".env"), "content": "K=V"}},
        cwd=tmp_path,
    )
    assert code == 0 and out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
