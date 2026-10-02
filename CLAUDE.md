# CLAUDE.md — analiticvolley

Sistema de Computer Vision para vóley. **MVP 1:** detección + tracking + identidad persistente + re-identificación
de jugadores, y tracking de pelota. **Fuera de alcance** hasta cerrar el MVP 1: análisis táctico, rotaciones,
detección de jugadas (saque/ataque/bloqueo/recepción), estadísticas, predicción.

Al usuario se le escribe en español rioplatense, claro y sin jerga innecesaria.

## Comandos

```bash
uv sync                      # core + dev (sin GPU). `uv sync --extra ml` agrega torch/ultralytics
uv run pytest                # todos los tests
uv run pytest tests/harness  # tests del harness (hooks y checks)
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Datos y videos: fuera del repo, en `VOLLEY_DATA_DIR` o `configs/local.yaml` (gitignored). Ver `docs/data/DATASETS.md`.

## Ciclo de vida obligatorio (detalle: docs/harness/WORKFLOW.md)

1. **Spike** (`/spike`) solo si hay incertidumbre técnica real; con alcance y criterio de fin. Si se omite, justificarlo en la spec.
2. **Spec** (`/spec`) en `docs/specs/NNN-*.md`. Debe pasar `scripts/check_spec.py`.
3. **ADR** (`/adr`) solo para decisiones arquitectónicas reales. Si no hace falta, decirlo en la spec.
4. **Issue** (`/issue`) por spec; alcance chico y verificable.
5. **TDD** (`/tdd`): test en rojo **por la razón esperada** → mínimo código → refactor.
6. **PR** (`/pr`) desde una rama `<issue>-<slug>`; cuerpo con `Closes #N` y la spec.
7. **CI verde y verificado** (qué corrió, no solo el color). El merge a `main` lo aprueba el humano.
8. **Informe de sprint** (`/sprint-report`).

## Reglas de producto (no negociables)

- `PLAYER ID ≠ TRACK ID`. Un track nuevo **nunca** crea un jugador automáticamente.
- La identidad combina equipo + tracking + contexto temporal + apariencia + Re-ID + dorsal. Ninguna señal sola es absoluta.
- El dorsal es metadata: si hay duda, `null`. Un número equivocado es peor que ninguno.
- Una predicción de pelota nunca se reporta como `DETECTED`.
- Ningún `lado = equipo` sin config que lo respalde.
- Sin rutas absolutas en código ni en configs versionadas.

## Reglas de ingeniería

- Ningún video, dataset ni archivo de datos entra al repo (es público). Lo controlan hooks y CI.
- No debilitar tests: nada de skip/xfail sin `#Issue`, ni borrar asserts para pasar CI.
- No declarar algo terminado sin evidencia (salida de tests, métricas, links de CI). Lo no medido es **desconocido**.
- Cambios mínimos y dentro del alcance del Issue. Sin dependencias nuevas fuera del plan aprobado.
- Separar en los informes: **hechos comprobados**, **hipótesis**, **decisiones tomadas**.
- No modificar `.claude/settings.json`, `.claude/hooks/`, `.github/workflows/`, `scripts/check_*.py` ni `quality/baseline.json` sin aprobación explícita (los hooks lo piden).

## Requiere aprobación humana explícita

Merge a `main`, borrar datos o ramas, operaciones irreversibles, cambios de permisos o controles,
dependencias nuevas, costos externos, cualquier cosa que exponga secretos.

## Mapa del repo

`src/volley_cv/` paquete · `tests/` (unit, scenarios, integration, harness) · `configs/` · `models/` (.onnx, ADR-0003) ·
`legacy/` (congelado) · `docs/{adr,specs,spikes,harness,data,sprints}` · `scripts/check_*.py` (quality gates) ·
`.claude/` (harness: settings, hooks, agents, skills).

Agentes: `architect` (spike/spec/ADR), `qa-engineer` (tests en rojo), `reviewer` (revisión independiente, solo lectura).
