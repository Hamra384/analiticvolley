# ADR-0001: Adoptar un AI Engineering Harness mínimo con guardrails como código

- **Estado:** aceptado
- **Fecha:** 2026-10-01
- **Issue:** #1, #2

## Contexto y problema
El desarrollo lo hace mayormente un agente (Claude Code) con un único humano que aprueba decisiones. El repo era
un conjunto de scripts sin tests, sin CI y sin trazabilidad. Las restricciones críticas (no subir videos, no
debilitar tests, no tocar `main` directo) no pueden depender solo de instrucciones al modelo.

## Decisión
Harness en capas, todo versionado en el repo:
1. Contexto: `CLAUDE.md` corto + `docs/harness/`.
2. Guardrails: permisos en `.claude/settings.json` + hooks Python testeados.
3. Workflow: skills (spike, spec, adr, issue, tdd, pr, sprint-report).
4. Roles: 3 subagentes (architect, qa-engineer, reviewer).
5. Verdad final: GitHub Actions + branch protection + CODEOWNERS.

## Alternativas evaluadas
| Alternativa | A favor | En contra |
|---|---|---|
| Solo `CLAUDE.md` | Simple | Las reglas no se hacen cumplir |
| 6 agentes (architect, dev, QA, security, reviewer, devops) | Separación fina | Costo y ruido sin beneficio demostrado en un proyecto de 1 persona |
| **Harness mínimo + CI** | Reglas verificables; crece con evidencia | Más archivos que mantener |

## Consecuencias
- Positivas: cada regla crítica tiene un control automatizado y un test.
- Negativas: los hooks agregan latencia a cada tool call (~100 ms por hook Python).

## Riesgos
Un hook no se protege a sí mismo por completo: la protección real es permiso `deny`/`ask` + CODEOWNERS +
branch protection.

## Criterio de revisión
Retro de cada sprint: falsos positivos/negativos de hooks, retrabajo, costo.
