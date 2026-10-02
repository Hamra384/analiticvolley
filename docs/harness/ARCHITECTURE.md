# Arquitectura del AI Engineering Harness

Decisión: [ADR-0001](../adr/0001-adoptar-ai-engineering-harness.md). Principio: **las reglas críticas se hacen
cumplir con controles automatizados**; las instrucciones al modelo son la última línea, no la primera.

```
Capa 5  Verdad final   GitHub Actions (ci.yml) + branch protection en main + CODEOWNERS
Capa 4  Trazabilidad   docs/specs → docs/adr → Issue → rama → commits → PR → CI   (check_traceability, check_spec)
Capa 3  Workflow       skills: /spike /spec /adr /issue /tdd /pr /sprint-report
Capa 2  Roles          subagentes: architect · qa-engineer · reviewer (solo lectura)
Capa 1  Guardrails     .claude/settings.json (permisos) + hooks Python testeados
Capa 0  Contexto       CLAUDE.md + docs/harness/
```

## Componentes

| Componente | Archivo | Qué garantiza | Test |
|---|---|---|---|
| guard_bash | `.claude/hooks/guard_bash.py` | Sin push forzado / a main, sin `--no-verify`, sin videos en `git add`, sin leer `.env`, sin borrar recursos remotos; pide aprobación para borrados recursivos, merges, dependencias | `tests/harness/test_guard_bash.py` |
| guard_edit | `.claude/hooks/guard_edit.py` | El agente no edita sus propios controles sin aprobación; sin videos/.env; sin debilitar tests | `tests/harness/test_guard_edit.py` |
| post_edit_lint | `.claude/hooks/post_edit_lint.py` | ruff fix+format en cada `.py` editado; errores no corregibles vuelven al agente | `tests/harness/test_post_hooks.py` |
| log_event | `.claude/hooks/log_event.py` | Telemetría JSONL sin contenido (solo metadatos) | `tests/harness/test_post_hooks.py` |
| check_no_media | `scripts/check_no_media.py` | Ningún video/dato/archivo grande versionado | `tests/harness/test_checks.py` |
| check_tests_ran | `scripts/check_tests_ran.py` | CI falla con 0 tests, < mínimo, fallos o skips sin Issue | ídem |
| check_spec | `scripts/check_spec.py` | Specs completas con AC verificables y Issue | ídem |
| check_traceability | `scripts/check_traceability.py` | PR → Issue; cambio de producto → spec | ídem |
| check_coverage_ratchet | `scripts/check_coverage_ratchet.py` | La cobertura no baja respecto de `quality/baseline.json` | ídem |
| harness_metrics | `scripts/harness_metrics.py` | Resumen de telemetría para el informe de sprint | `tests/harness/test_harness_metrics.py` |

Los hooks usan solo la stdlib (corren con el `python` del sistema, fuera del venv) y se prueban en Ubuntu y
Windows en CI.

## Límites conocidos (honestos)

- **El agente usa la cuenta de GitHub del dueño.** GitHub no puede exigir una revisión de alguien distinto al
  autor; CODEOWNERS es informativo. El checkpoint humano del merge lo implementa `guard_bash` (`gh pr merge` → ask).
- Un hook no se protege a sí mismo por completo: si alguien desactiva los hooks en la configuración local, solo
  quedan CI y branch protection.
- Los hooks inspeccionan el texto del comando; un comando ofuscado (por ejemplo, armado dentro de un script)
  puede evadirlos. CI y branch protection son la red de seguridad.
- Si el hook falla al ejecutarse (por ejemplo, `python` no está en PATH), Claude Code lo trata como error no
  bloqueante. Ver RUNBOOK.md → "Verificar que los hooks están activos".
