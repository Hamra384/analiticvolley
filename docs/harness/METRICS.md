# Métricas y quality gates

Regla: **lo no medido se reporta como desconocido, nunca como aprobado.** Los umbrales iniciales se validan con
el baseline; si alguno resulta irrealista, se documenta la evidencia antes de cambiarlo.

## Gates automatizados (CI, bloquean el merge)

| Gate | Umbral | Justificación |
|---|---|---|
| Tests ejecutados | ≥ 100 en ubuntu y en el job del harness; 0 fallos | Detecta colapso de recolección (pipeline verde con 0 tests) |
| Tests salteados | 0 sin referencia `#Issue` | Un skip sin dueño es un test desactivado |
| Cobertura de líneas (`volley_cv`) | ≥ baseline − 0,5 pt (`quality/baseline.json`) | Trinquete: no se fija un % arbitrario, solo se impide empeorar |
| Lint / formato / tipos | 0 errores (ruff, mypy strict) | Estándar |
| Medios en el repo | 0 | Repo público |
| Secretos | 0 hallazgos de gitleaks | Repo público |
| Vulnerabilidades de dependencias | 0 reportadas por pip-audit (core + dev) | Estándar |
| Specs | todas completas con AC verificables | SDD |
| Trazabilidad | PR → Issue; producto → spec | Git como fuente de verdad |

## Métricas del producto (MVP 1) — objetivos iniciales a validar con baseline

Se miden sobre los 10 clips de `configs/eval/clips.yaml` (evaluación local con GPU, no en CI).

| Métrica | Objetivo inicial | Por qué importa |
|---|---|---|
| IDF1 (jugadores en cancha) | ≥ 0,80 | Métrica principal de identidad persistente |
| ID switches | 0 en clips de superposición; ≤ 2 / 1000 frames | El error central del MVP |
| ID consistency | ≥ 0,95 | Cada jugador real ↔ una sola ID |
| Occlusion recovery rate | ≥ 0,90 | Requisito de oclusión |
| Re-ID accuracy | ≥ 0,90 | Asociación tras pérdida |
| Track fragmentation | se reporta | Separa culpas tracker vs. IdentityManager |
| Recall de detección en cancha | ≥ 0,95 | Techo de todo lo demás |
| Precisión del dorsal asignado | ≥ 0,95 (preferir null) | Un número mal asignado contamina la identidad |
| Pelota recall / precision | ≥ 0,80 / ≥ 0,90 | Detección base |
| Error de posición de pelota | mediana ≤ 10 px | Utilidad de la predicción |
| Continuidad / re-adquisición de pelota | ≥ 0,90 | "Que no desaparezca" |
| Pelotas falsas | ≤ 1 por clip | Sin pelotas duplicadas |
| FPS | se reporta (meta ≥ 15 en GTX 1660 Super) | Tiempo cercano al real |
| Identidad a través de cortes | se reporta (secundaria) | Edición solo-rallies |

## Métricas del proceso (informe de sprint)

| Métrica | Fuente |
|---|---|
| Tasa de fallos de CI | `gh run list` |
| Defectos encontrados antes / después de integrar | Issues `type:bug` vs. hallazgos del reviewer |
| Retrabajo | commits de fix sobre el mismo Issue; PRs reabiertos |
| Tiempo de ciclo | apertura de Issue → merge |
| Bloqueos de hooks (falsos positivos / negativos) | `scripts/harness_metrics.py` + retro |
| Tasa de error de herramientas | `scripts/harness_metrics.py` |
| Costo LLM | `/cost` de Claude Code por sesión (no hay API de costos en el harness: estimado) |
| Deuda técnica | Issues con label `debt` |
