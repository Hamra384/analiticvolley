# SPEC-002: Pipeline integrado de jugadores y video de debug

- **Issue:** #8
- **Estado:** aprobada (autorización general del usuario para S3–S4, 2026-10-02)
- **ADR relacionados:** ADR-0002 (arquitectura). No requiere ADR nuevo: implementa las capas ya decididas.
- **Spike:** SPIKE-001 (tracker ByteTrack). La pelota queda fuera (S5, SPIKE-002).

## Problema
Los componentes existen por separado (detector en spikes, `IdentityManager` puro). Falta el pipeline que va de un
video a la salida por frame del punto 13 del MVP, y la visualización de debug (punto 14), imprescindible para ver
errores de identidad.

## Objetivos y alcance
- Dentro: lectura de clip, detección de cortes de edición, detección de personas, filtro de cancha, tracking
  ByteTrack, embedding de apariencia, clasificación de equipo por configuración de colores (incluye líberos),
  `IdentityManager`, salida JSONL validada, video de debug, CLI.
- Fuera: pelota (S5), dorsales/OCR (S5; la interfaz acepta lecturas, hoy ninguna), evaluación con ground truth (S6).

## Requerimientos funcionales
- RF-1: `ShotDetector` en streaming: corte = diferencia media absoluta del frame (gris 160×90) contra el anterior
  > `ratio` × mediana móvil (61 frames) y > `min_diff` (valores calibrados en la auditoría: 6 y 3). En un corte se
  llama `IdentityManager.reset()`.
- RF-2: `CourtMask`: una persona está "en cancha" si su punto de apoyo (centro inferior de la caja) cae sobre la
  máscara de color de la cancha (rango HSV por video, en `configs/videos/<video>.yaml`), dilatada `margin_px`.
  Solo las personas en cancha pasan al tracker. Se excluyen regiones configuradas (marcador sobreimpreso).
- RF-3: `TeamClassifier`: color de torso (mediana Lab) → prototipo más cercano entre los configurados
  (equipo A, equipo B, líbero A, líbero B). Si la distancia supera `max_dist` → equipo desconocido. Si un color es
  ambiguo entre el principal de un equipo y el líbero del otro (diferencia de distancias < `ambiguity_margin`),
  se resuelve por el lado de la red: dentro del tramo, el lado de cada equipo se estima con la mediana del punto de
  apoyo de sus jugadores no ambiguos. Sin esa evidencia → desconocido (nunca se asume lado = equipo sin datos).
- RF-4: Embeddings de apariencia sobre el recorte de la caja; normalizados. **Decisión de implementación (S4):**
  ResNet18 ImageNet (torchvision, BSD) en lugar de OSNet x0.25, porque la API de Re-ID de boxmot 25 cambió y
  resolverla requiere un spike. Es reversible (interfaz `Embedder`); se compara contra OSNet en S6 con datos
  anotados.
- RF-5: Salida JSONL: una línea `FrameOutput` por frame procesado, con el índice de frame absoluto del video.
- RF-6: Video de debug: caja coloreada por equipo, etiqueta con `player_id` corto (p. ej. `A04`), dorsal si existe,
  estado y `track_id` (opcional), trayectoria de los últimos `trail` frames, línea de marcador del frame/estado
  global, y marca de corte de edición. Cada elemento se puede activar/desactivar.
- RF-7: CLI: `python -m volley_cv run --clip A2 [--out DIR] [--no-video]` y `--video PATH --start mm:ss --end mm:ss`.
- RF-8: Los componentes con modelos (detector, tracker, Re-ID) se inyectan detrás de interfaces (`Protocol`),
  de modo que el pipeline se testea en CI con implementaciones falsas, sin GPU.

## Requerimientos no funcionales
- RNF-1: Sin rutas absolutas: videos vía `volley_cv.config`; salidas en `<data>/outputs/` por defecto.
- RNF-2: Reporta FPS de punta a punta al terminar (se registra, no es gate).
- RNF-3: Determinismo dado el mismo video y modelos (sin aleatoriedad propia).

## Casos de uso
- CU-1: `python -m volley_cv run --clip A2` → `<data>/outputs/A2/frames.jsonl` + `debug.mp4`.
- CU-2: Tests de integración con detector/tracker/embedder falsos sobre frames sintéticos.

## Reglas de negocio
- RN-1: Ningún "lado = equipo" fijo: el lado se estima por tramo y solo desambigua colores ambiguos.
- RN-2: Personas fuera de la cancha (banco, oficiales, público) no generan identidades de jugador.

## Criterios de aceptación
- [ ] AC-1: dado una secuencia de frames con un cambio brusco de imagen, cuando la procesa el `ShotDetector`, entonces marca exactamente ese frame como corte y no marca cortes en un paneo suave.
- [ ] AC-2: dado un frame con una zona naranja de cancha y dos personas, una con el apoyo dentro y otra fuera, cuando se aplica `CourtMask`, entonces solo pasa la de adentro; y una persona dentro de una región excluida no pasa.
- [ ] AC-3: dado prototipos de color de equipos y líberos, cuando se clasifica un torso con el color del equipo A, del equipo B o de un líbero no ambiguo, entonces devuelve el equipo correcto; y un color lejano a todos devuelve desconocido.
- [ ] AC-4: dado un líbero cuyo color coincide con el principal del otro equipo, cuando hay evidencia de lado en el tramo, entonces se clasifica en el equipo de su lado; y sin evidencia de lado, entonces devuelve desconocido.
- [ ] AC-5: dado un pipeline con detector/tracker/embedder falsos sobre frames sintéticos, cuando se corre, entonces escribe una línea JSONL por frame, cada una valida contra `FrameOutput`, y los `player_id` son estables.
- [ ] AC-6: dado un corte de edición en la secuencia sintética, cuando corre el pipeline, entonces se llama `reset()` del `IdentityManager` en ese frame.
- [ ] AC-7: dado un frame y su `FrameOutput`, cuando se dibuja el debug, entonces la imagen cambia en las cajas de los jugadores, el color depende del equipo y la etiqueta incluye el `player_id` corto; con todo desactivado, la imagen no cambia.
- [ ] AC-8: dado el clip A2 real (GPU, local), cuando se corre el CLI, entonces produce `frames.jsonl` válido con 600 líneas y `debug.mp4` reproducible (evidencia manual registrada en el informe; no corre en CI).

## Restricciones técnicas
Python 3.12; ultralytics, boxmot, torch solo en el extra `ml` e importados de forma diferida. OpenCV para video.

## Manejo de errores
- Video inexistente o ilegible: error claro con la ruta resuelta.
- Configuración de video ausente: error claro indicando el archivo esperado en `configs/videos/`.
- Frame ilegible en mitad del clip: se termina el clip y se registra cuántos frames se procesaron.
- Recorte vacío para Re-ID/torso: la observación sigue sin embedding / sin equipo.

## Seguridad
Sin datos sensibles. Los videos no se copian al repo; las salidas van al directorio de datos (fuera del repo).

## Estrategia de testing
| AC | Test |
|---|---|
| AC-1 | `tests/unit/test_shots.py` (frames sintéticos) |
| AC-2 | `tests/unit/test_court.py` (imagen sintética) |
| AC-3, AC-4 | `tests/unit/test_team.py` (colores sintéticos) |
| AC-5, AC-6 | `tests/integration/test_pipeline.py` (fakes, sin GPU) |
| AC-7 | `tests/unit/test_viz.py` |
| AC-8 | ejecución manual local sobre A2; evidencia en `docs/sprints/S4.md` |

## Dependencias e impacto
Nuevos módulos `volley_cv.{video,court,team,detection,tracking,appearance,viz,pipeline,cli}`. Sin dependencias
nuevas (las de modelos ya están en el extra `ml`). Consume SPEC-001.
