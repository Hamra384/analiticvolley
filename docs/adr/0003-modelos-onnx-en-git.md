# ADR-0003: Mantener los modelos `.onnx` en git

- **Estado:** aceptado
- **Fecha:** 2026-10-01
- **Issue:** #1

## Contexto y problema
Hay dos modelos entrenados versionados (`digit_classifier.onnx` 2,6 MB, `dorsal_classifier.onnx` 6,2 MB). Se
propuso moverlos a GitHub Releases; el usuario decidió mantenerlos en git.

## Decisión
Los modelos `.onnx` viven en `models/` dentro del repo. Los pesos `.pt` descargables (YOLO) siguen ignorados.

## Alternativas evaluadas
| Alternativa | A favor | En contra |
|---|---|---|
| **Git** | Simple; reproducible por commit | Infla el historial si se reentrenan seguido |
| GitHub Releases + checksum | Historial liviano | Paso extra de descarga |
| Git LFS | Historial liviano | Cuota de LFS; complejidad |

## Consecuencias
- Positivas: un `git clone` alcanza para tener los modelos.
- Negativas: cada reentrenamiento suma ~3–6 MB al historial.

## Criterio de revisión
Si `models/` supera 50 MB acumulados en el historial, se reabre (LFS o Releases).
Nota: ninguno de los dos modelos se usa en el pipeline legacy; su utilidad se evalúa en S5.
