---
name: sprint-report
description: Genera el informe consolidado de fin de sprint en docs/sprints/ con evidencia verificable, métricas del harness y preguntas agrupadas para el humano.
---

1. Recolectá evidencia (no de memoria):
   - `git log --oneline main..HEAD` o del sprint; `gh issue list --label sprint:SN --state all`; `gh pr list --state all`.
   - Último run de CI: `gh run list -L 5` y `gh run view <id> --log | grep check_` (qué ejecutó realmente).
   - Telemetría: `uv run python scripts/harness_metrics.py` (si existe) o `.harness/logs/events.jsonl`.
2. Escribí `docs/sprints/SN.md` con: objetivos alcanzados · archivos creados/modificados · Issues y ADR ·
   pruebas ejecutadas y resultados · estado de GitHub Actions · errores detectados y corregidos ·
   problemas pendientes · decisiones tomadas · riesgos · costos estimados · retro del harness
   (falsos positivos/negativos, retrabajo) · dudas para el humano · próximas tareas.
3. Separá **hechos comprobados**, **hipótesis** y **decisiones**. Lo no medido se reporta como desconocido.
4. Al humano: resumen corto en lenguaje simple + link al archivo. Preguntas agrupadas al final.
