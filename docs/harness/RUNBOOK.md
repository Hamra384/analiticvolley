# Runbook

## Setup de una máquina nueva

```bash
git clone https://github.com/Hamra384/analiticvolley && cd analiticvolley
uv sync                                     # core + dev
echo "data_dir: <ruta a datos>" > configs/local.yaml
uv run pytest                               # debe pasar todo
```

Los hooks necesitan `python` (3.10+) en el PATH del sistema (solo usan la stdlib).

## Verificar que los hooks están activos

Un hook que no puede ejecutarse falla en silencio (error no bloqueante). Después de cambiar
`.claude/settings.json` o de instalar en otra máquina, pedile al agente que intente `git push --force` en una rama
de prueba: debe aparecer `[harness:guard_bash] push forzado…`. Si no aparece, revisar `python` en PATH y `/hooks`.

## CI falla

1. `gh run view <id> --log-failed`.
2. Si falla `check_tests_ran`: mirar cuántos tests corrieron y qué se salteó.
3. Si falla el trinquete de cobertura: agregar tests. Bajar el baseline requiere un PR explícito con justificación.
4. Si falla pip-audit: actualizar la dependencia afectada (con aprobación) o documentar la excepción en un Issue.

## Recuperación

- Revertir un merge: `git revert -m 1 <merge-commit>` en una rama nueva + PR. Nunca `push --force` a `main`.
- Commit equivocado en una rama de trabajo propia aún no compartida: `git reset --soft HEAD~1` (pide aprobación si es `--hard`).
- Video agregado por error: si no se pusheó, `git rm --cached <archivo>`; si se pusheó, **avisar al humano**
  (requiere reescritura de historial → autorización explícita).

## Incidentes y lecciones aprendidas

Registrar en `docs/sprints/SN.md` → sección "Retro del harness".
