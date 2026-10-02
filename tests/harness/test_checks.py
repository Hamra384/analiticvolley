from pathlib import Path
from types import ModuleType

import pytest

from .conftest import SCRIPTS, load


@pytest.fixture(scope="module")
def no_media() -> ModuleType:
    return load(SCRIPTS / "check_no_media.py")


@pytest.fixture(scope="module")
def tests_ran() -> ModuleType:
    return load(SCRIPTS / "check_tests_ran.py")


@pytest.fixture(scope="module")
def spec() -> ModuleType:
    return load(SCRIPTS / "check_spec.py")


@pytest.fixture(scope="module")
def trace() -> ModuleType:
    return load(SCRIPTS / "check_traceability.py")


@pytest.fixture(scope="module")
def ratchet() -> ModuleType:
    return load(SCRIPTS / "check_coverage_ratchet.py")


# ── check_no_media ─────────────────────────────────────────────────────────────
class TestNoMedia:
    def test_flags_videos_and_data_dirs(self, no_media: ModuleType) -> None:
        files = {"src/a.py": 10, "videos/x.mp4": 10, "clip.MOV": 5, "data/frames/1.jpg": 3}
        bad = no_media.violations(files)
        assert {b.split(":")[0] for b in bad} == {"videos/x.mp4", "clip.MOV", "data/frames/1.jpg"}

    def test_flags_large_files_but_allows_models(self, no_media: ModuleType) -> None:
        mb = 1024 * 1024
        files = {"big.bin": 11 * mb, "models/dorsal_classifier.onnx": 7 * mb, "models/huge.onnx": 30 * mb}
        bad = no_media.violations(files)
        assert {b.split(":")[0] for b in bad} == {"big.bin", "models/huge.onnx"}

    def test_clean_tree_passes(self, no_media: ModuleType) -> None:
        assert no_media.violations({"README.md": 100, "configs/eval/clips.yaml": 200}) == []

    def test_current_repo_is_clean(self, no_media: ModuleType) -> None:
        assert no_media.main([]) == 0


# ── check_tests_ran ────────────────────────────────────────────────────────────
JUNIT = """<?xml version="1.0"?><testsuites>
<testsuite name="pytest" tests="{tests}" failures="{failures}" errors="0" skipped="{skipped}">
{cases}
</testsuite></testsuites>"""


def junit(tmp_path: Path, tests: int, failures: int = 0, skips: tuple[str, ...] = ()) -> Path:
    cases = "".join(f'<testcase name="t{i}"><skipped message="{m}"/></testcase>' for i, m in enumerate(skips))
    p = tmp_path / "junit.xml"
    p.write_text(
        JUNIT.format(tests=tests, failures=failures, skipped=len(skips), cases=cases), encoding="utf-8"
    )
    return p


class TestTestsRan:
    def test_zero_tests_fails(self, tests_ran: ModuleType, tmp_path: Path) -> None:
        assert tests_ran.main([str(junit(tmp_path, 0))]) == 1

    def test_failures_fail(self, tests_ran: ModuleType, tmp_path: Path) -> None:
        assert tests_ran.main([str(junit(tmp_path, 5, failures=1))]) == 1

    def test_unjustified_skip_fails(self, tests_ran: ModuleType, tmp_path: Path) -> None:
        assert tests_ran.main([str(junit(tmp_path, 5, skips=("later",)))]) == 1

    def test_skip_with_issue_reference_passes(self, tests_ran: ModuleType, tmp_path: Path) -> None:
        assert tests_ran.main([str(junit(tmp_path, 5, skips=("requiere uv (#2)",)))]) == 0

    def test_below_minimum_fails(self, tests_ran: ModuleType, tmp_path: Path) -> None:
        assert tests_ran.main([str(junit(tmp_path, 3)), "--min-tests", "10"]) == 1

    def test_missing_report_fails(self, tests_ran: ModuleType, tmp_path: Path) -> None:
        assert tests_ran.main([str(tmp_path / "nope.xml")]) == 1


# ── check_spec ─────────────────────────────────────────────────────────────────
GOOD_SPEC = """# SPEC-001: Algo
- **Issue:** #3
## Problema
x
## Objetivos y alcance
x
## Requerimientos funcionales
x
## Requerimientos no funcionales
x
## Criterios de aceptación
- [ ] AC-1: dado X, cuando Y, entonces Z
## Manejo de errores
x
## Seguridad
x
## Estrategia de testing
x
## Dependencias e impacto
x
"""


class TestSpec:
    def test_complete_spec_passes(self, spec: ModuleType) -> None:
        assert spec.problems(GOOD_SPEC) == []

    def test_missing_section_is_reported(self, spec: ModuleType) -> None:
        text = GOOD_SPEC.replace("## Seguridad\nx\n", "")
        assert any("Seguridad" in p for p in spec.problems(text))

    def test_acceptance_criteria_must_be_checkable(self, spec: ModuleType) -> None:
        text = GOOD_SPEC.replace("- [ ] AC-1: dado X, cuando Y, entonces Z", "Que funcione bien.")
        assert any("criterio" in p.lower() for p in spec.problems(text))

    def test_issue_reference_required(self, spec: ModuleType) -> None:
        assert any("Issue" in p for p in spec.problems(GOOD_SPEC.replace("#3", "pendiente")))

    def test_template_is_skipped_and_repo_specs_are_valid(self, spec: ModuleType) -> None:
        assert spec.main([]) == 0


# ── check_traceability ─────────────────────────────────────────────────────────
class TestTraceability:
    def test_pr_without_issue_fails(self, trace: ModuleType) -> None:
        assert trace.problems("Arreglé cosas", ["README.md"])

    @pytest.mark.parametrize("kw", ["Closes #4", "fixes #4", "Refs #12", "Resolves #1"])
    def test_issue_reference_passes(self, trace: ModuleType, kw: str) -> None:
        assert trace.problems(f"Cambio.\n\n{kw}", ["README.md"]) == []

    def test_src_change_requires_spec(self, trace: ModuleType) -> None:
        assert trace.problems("Closes #4", ["src/volley_cv/identity/manager.py"])

    def test_src_change_with_spec_passes(self, trace: ModuleType) -> None:
        body = "Closes #4\nSpec: docs/specs/001-identity.md"
        assert trace.problems(body, ["src/volley_cv/identity/manager.py"]) == []

    def test_spec_na_needs_reason(self, trace: ModuleType) -> None:
        assert trace.problems("Closes #4\nSpec: N/A", ["src/volley_cv/config.py"])
        assert (
            trace.problems(
                "Closes #4\nSpec: N/A (bootstrap de config, sin lógica de producto)",
                ["src/volley_cv/config.py"],
            )
            == []
        )


# ── check_coverage_ratchet ─────────────────────────────────────────────────────
COV = '<?xml version="1.0" ?><coverage line-rate="{rate}" branch-rate="0.5"></coverage>'


def cov(tmp_path: Path, rate: float) -> Path:
    p = tmp_path / "coverage.xml"
    p.write_text(COV.format(rate=rate), encoding="utf-8")
    return p


class TestRatchet:
    def test_drop_below_baseline_fails(self, ratchet: ModuleType, tmp_path: Path) -> None:
        base = tmp_path / "baseline.json"
        base.write_text('{"coverage_line_rate": 0.90}', encoding="utf-8")
        assert ratchet.main([str(cov(tmp_path, 0.85)), "--baseline", str(base)]) == 1

    def test_within_tolerance_passes(self, ratchet: ModuleType, tmp_path: Path) -> None:
        base = tmp_path / "baseline.json"
        base.write_text('{"coverage_line_rate": 0.90}', encoding="utf-8")
        assert ratchet.main([str(cov(tmp_path, 0.897)), "--baseline", str(base)]) == 0

    def test_missing_baseline_fails_unless_bootstrapping(self, ratchet: ModuleType, tmp_path: Path) -> None:
        args = [str(cov(tmp_path, 0.8)), "--baseline", str(tmp_path / "none.json")]
        assert ratchet.main(args) == 1

    def test_update_writes_baseline(self, ratchet: ModuleType, tmp_path: Path) -> None:
        base = tmp_path / "baseline.json"
        assert ratchet.main([str(cov(tmp_path, 0.77)), "--baseline", str(base), "--update"]) == 0
        assert "0.77" in base.read_text(encoding="utf-8")
