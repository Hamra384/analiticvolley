from pathlib import Path
from types import ModuleType

import pytest

from .conftest import run_hook

DENY = [
    # historial compartido / protección de main
    "git push --force origin feature",
    "git push -f",
    "git push --force-with-lease origin s1/harness",
    "git push origin main",
    "git push origin HEAD:main",
    "git push upstream master",
    # saltear controles
    "git commit --no-verify -m wip",
    "git commit -n -m wip",
    "git commit --no-gpg-sign -m x",
    "claude --dangerously-skip-permissions",
    "claude --permission-mode bypassPermissions",
    "gh pr merge 3 --admin",
    # medios al repo
    "git add videos/partido.mp4",
    "git add clip.MKV other.py",
    "git add -A data/",
    # secretos
    "cat .env",
    "Get-Content .env.local",
    "type .env",
    # irreversibles remotos
    "gh repo delete Hamra384/analiticvolley --yes",
    "gh api -X DELETE repos/Hamra384/analiticvolley",
    "gh release delete v1 -y",
]

ASK = [
    "git reset --hard HEAD~1",
    "git clean -fdx",
    "rm -rf build",
    "rm -fr /tmp/x",
    "Remove-Item -Recurse -Force outputs",
    "git add -f ignored.txt",
    "gh pr merge 4 --squash",
    "gh api -X PUT repos/Hamra384/analiticvolley/branches/main/protection --input p.json",
    "git branch -D old",
    "git push origin --delete s0/bootstrap",
    "uv add torch",
    "pip install requests",
]

ALLOW = [
    "git status",
    "git push -u origin s1/harness",
    "git push origin s1/harness",
    "git commit -m 'S1: hooks (Refs #2)'",
    "git add src/volley_cv/config.py tests/",
    "git log --oneline -5",
    "uv run pytest -q",
    "uv sync",
    "gh pr create --title x --body y",
    "rm build/tmp.txt",
    "python scripts/check_no_media.py",
    "echo 'the .env file is ignored'",
    "grep -r mp4 docs/",
    # falso positivo real (S1): un mensaje de commit que menciona la opción no es usarla
    "git commit -m 'settings: disableBypassPermissionsMode'",
    "git commit -q -m @'\nS1\n- settings (permisos, disableBypassPermissionsMode, hooks)\n'@",
]


@pytest.mark.parametrize("cmd", DENY)
def test_denies(guard_bash: ModuleType, cmd: str) -> None:
    d = guard_bash.evaluate(cmd, current_branch="s1/harness")
    assert d is not None and d.decision == "deny", cmd


@pytest.mark.parametrize("cmd", ASK)
def test_asks(guard_bash: ModuleType, cmd: str) -> None:
    d = guard_bash.evaluate(cmd, current_branch="s1/harness")
    assert d is not None and d.decision == "ask", cmd


@pytest.mark.parametrize("cmd", ALLOW)
def test_allows(guard_bash: ModuleType, cmd: str) -> None:
    assert guard_bash.evaluate(cmd, current_branch="s1/harness") is None, cmd


def test_bare_push_on_main_is_denied(guard_bash: ModuleType) -> None:
    d = guard_bash.evaluate("git push", current_branch="main")
    assert d is not None and d.decision == "deny"


def test_chained_commands_are_each_checked(guard_bash: ModuleType) -> None:
    d = guard_bash.evaluate("uv run pytest && git push --force", current_branch="x")
    assert d is not None and d.decision == "deny"


def test_deny_wins_over_ask(guard_bash: ModuleType) -> None:
    d = guard_bash.evaluate("git reset --hard; git push -f", current_branch="x")
    assert d is not None and d.decision == "deny"


def test_protocol_end_to_end(tmp_path: Path) -> None:
    code, out, _ = run_hook(
        "guard_bash.py",
        {"tool_name": "Bash", "tool_input": {"command": "git push --force"}, "session_id": "t"},
        cwd=tmp_path,
    )
    assert code == 0
    assert out is not None
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert hso["permissionDecisionReason"]


def test_protocol_allows_silently(tmp_path: Path) -> None:
    code, out, _ = run_hook(
        "guard_bash.py", {"tool_name": "PowerShell", "tool_input": {"command": "git status"}}, cwd=tmp_path
    )
    assert code == 0 and out is None


def test_malformed_input_fails_closed(tmp_path: Path) -> None:
    """Si el hook no puede leer la entrada, no debe dejar pasar el comando en silencio."""
    code, out, _ = run_hook("guard_bash.py", {"tool_name": "Bash", "tool_input": {}}, cwd=tmp_path)
    assert code == 0 and out is not None
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_decisions_are_logged_without_command_text(tmp_path: Path) -> None:
    run_hook("guard_bash.py", {"tool_name": "Bash", "tool_input": {"command": "cat .env"}}, cwd=tmp_path)
    log = (tmp_path / ".harness-test-logs" / "events.jsonl").read_text(encoding="utf-8")
    assert '"decision": "deny"' in log
    assert ".env" not in log  # no se registra el comando (puede contener secretos)
