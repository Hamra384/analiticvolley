# Política de permisos

Fuente de verdad: `.claude/settings.json` (permisos + hooks) y `.claude/hooks/`. Este documento explica el porqué.

## Sin consulta (autonomía dentro del alcance aprobado)

Leer y analizar el repo · crear y modificar código, tests y docs · ejecutar `uv run …` (tests, linters,
scripts) · crear ramas y commits locales · push de ramas de trabajo · crear Issues y PRs.

## Requiere aprobación humana (`ask`)

| Acción | Control |
|---|---|
| Merge a `main` (`gh pr merge`) | guard_bash |
| Cambiar branch protection | guard_bash |
| Borrados recursivos, `git reset --hard`, `git clean -f`, `git branch -D`, borrar ramas remotas | guard_bash |
| Instalar dependencias (`uv add`, `pip install`, `npm install`) | guard_bash |
| `git add -f` (forzar archivos ignorados) | guard_bash |
| Editar `.claude/settings.json`, `.claude/hooks/**`, `.github/workflows/**`, `.github/CODEOWNERS`, `scripts/check_*.py`, `quality/baseline.json` | guard_edit + permiso `ask` |
| Agregar skip/xfail sin `#Issue`, quitar asserts o funciones de test | guard_edit |

## Prohibido (`deny`)

| Acción | Control |
|---|---|
| Push forzado / refspec `+` / `--mirror` | guard_bash + permiso deny |
| Push directo a `main` | guard_bash + branch protection |
| `git commit --no-verify` / `--no-gpg-sign` | guard_bash |
| `gh pr merge --admin` | guard_bash |
| Agregar videos o directorios de datos a git | guard_bash, guard_edit, check_no_media (CI) |
| Leer o escribir `.env*` | guard_bash, guard_edit, permiso deny |
| `gh repo delete`, `gh release delete`, `gh api -X DELETE` | guard_bash |
| `bypassPermissions` / `--dangerously-skip-permissions` | guard_bash + `disableBypassPermissionsMode` |

## Cambios a esta política

Siguen el ciclo normal (Issue → PR) y requieren aprobación humana explícita. El agente no relaja sus propias
restricciones para facilitar su ejecución.
