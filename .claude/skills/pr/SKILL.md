---
name: pr
description: Prepara y abre un Pull Request trazable (Issue, spec, tests ejecutados, riesgos), con revisión independiente previa y verificación de qué ejecutó CI.
---

1. Verificación local: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
   `uv run python scripts/check_no_media.py`. Todo en verde; si no, se corrige antes.
2. Revisión independiente: invocá al subagente `reviewer` con el diff (`git diff main...HEAD`). Resolvé los
   hallazgos bloqueantes o justificá por qué no.
3. `git push -u origin <rama>` y `gh pr create` usando `.github/pull_request_template.md`. El cuerpo debe tener
   `Closes #N` y la spec (o `Spec: N/A (<motivo>)`).
4. Esperá CI y **verificá qué corrió**: `gh pr checks`, y en el log del job de tests la línea de
   `check_tests_ran` (cantidad de tests, fallos, salteados). Un verde con 0 tests no es verde.
5. El merge a `main` lo aprueba el humano (el hook pide confirmación).
