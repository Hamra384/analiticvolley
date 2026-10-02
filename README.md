# analiticvolley

Sistema de Computer Vision para análisis de partidos de vóley.

**MVP 1 (en curso):** detección, tracking y re-identificación persistente de jugadores, y tracking de pelota.
Fuera de alcance por ahora: análisis táctico, estadísticas, detección de jugadas.

## Setup

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
uv sync                 # core + herramientas de desarrollo (sin GPU)
uv sync --extra ml      # + torch/ultralytics (Windows: CUDA 12.1)
```

Los videos y datos **no** viven en el repo. Indicá dónde están:

```bash
# opción 1: variable de entorno
export VOLLEY_DATA_DIR=/ruta/a/datos
# opción 2: configs/local.yaml (gitignored)
echo "data_dir: D:/AIVolley/analiticvolley" > configs/local.yaml
```

## Comandos

```bash
uv run pytest                     # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## Estructura

| Ruta | Contenido |
|---|---|
| `src/volley_cv/` | Paquete principal |
| `tests/` | Tests (unit, scenarios, integration, harness) |
| `configs/` | Configuración versionada (clips de evaluación, videos) |
| `models/` | Modelos `.onnx` versionados (ver ADR-0003) |
| `legacy/` | Scripts originales, congelados como referencia (ver `legacy/README.md`) |
| `docs/` | ADRs, specs, spikes, harness, datos |
| `.claude/` | Configuración del AI Engineering Harness |

Flujo de trabajo y reglas: ver [CLAUDE.md](CLAUDE.md) y [docs/harness/WORKFLOW.md](docs/harness/WORKFLOW.md).
