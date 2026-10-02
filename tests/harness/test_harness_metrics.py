from .conftest import SCRIPTS, load

hm = load(SCRIPTS / "harness_metrics.py")


def test_summarize_counts_tools_errors_and_guards() -> None:
    records = [
        {"event": "PostToolUse", "tool": "Bash", "session": "a", "error": False},
        {"event": "PostToolUse", "tool": "Bash", "session": "a", "error": True},
        {"event": "PostToolUse", "tool": "Edit", "session": "b", "error": False},
        {"event": "guard_decision", "decision": "deny", "reason": "push a main", "session": "a"},
        {"event": "lint_block", "session": "b"},
    ]
    s = hm.summarize(records)
    assert s["sessions"] == 2
    assert s["tool_uses"] == 3
    assert s["tool_error_rate"] == 0.333
    assert s["tools"] == {"Bash": 2, "Edit": 1}
    assert s["guard_decisions"] == {"deny: push a main": 1}
    assert s["lint_blocks"] == 1


def test_no_tool_uses_reports_unknown_error_rate() -> None:
    assert hm.summarize([])["tool_error_rate"] is None
