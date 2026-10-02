# SPEC-001: IdentityManager — identidad persistente de jugadores

- **Issue:** #7
- **Estado:** aprobada (autorización general del usuario para S3–S4, 2026-10-02)
- **ADR relacionados:** ADR-0002 (tracking corto + identidad persistente; ByteTrack; partición de tracklets)
- **Spike:** SPIKE-001 (motiva la partición de tracklets). No hace falta spike nuevo: el diseño es lógica de
  asociación determinista, verificable con escenarios sintéticos.

## Problema
El tracker (ByteTrack) entrega IDs de corto plazo que se cortan en oclusiones (~5 IDs por jugador en 20 s) y que
además se **intercambian** entre jugadores (~1 vez por track largo, SPIKE-001). Usar el `track_id` como identidad
crea jugadores falsos y mezcla identidades. El MVP exige `PLAYER ID ≠ TRACK ID`, identidad estable sin depender del
número de camiseta, y número como metadata.

## Objetivos y alcance
- Dentro: componente puro (sin video, sin GPU) que recibe observaciones por frame (caja, track del tracker, equipo,
  embedding de apariencia, lectura de dorsal opcional) y emite el estado de identidad por frame.
  Incluye: partición de tracklets, asociación tracklet → identidad, máquina de estados, votación de dorsal,
  sustituciones, esquema de salida.
- Fuera: detección, tracking, cálculo de embeddings, clasificación de equipo, OCR (son entradas; S4/S5);
  pelota (SPEC aparte, S5); identidad a través de cortes de edición (secundaria).

## Requerimientos funcionales
- RF-1: Cada jugador recibe un `player_id` con formato `TEAM_{A|B}_PLAYER_{NN}` (NN correlativo por equipo, desde 01).
  El ID no codifica el dorsal.
- RF-2: El `track_id` de salida es el del tracklet vigente (`track_<id>` o `track_<id>.<n>` si fue partido).
- RF-3: Un tracklet nuevo **no** crea una identidad hasta acumular `confirm_frames` observaciones; antes se intenta
  asociarlo a identidades existentes no vigentes (OCCLUDED/LOST) del mismo equipo.
- RF-4: Asociación por costo combinado (algoritmo húngaro): apariencia (distancia coseno a la galería de la
  identidad), movimiento (distancia a la posición predicha, normalizada por un radio que crece con el tiempo
  perdido, con tope por velocidad máxima), dorsal (bonificación si coincide con confianza; penalización si
  contradice; nunca veto absoluto) y equipo (condición dura).
- RF-4b (revisión S3): sin embedding de apariencia, un tracklet necesita ≥ `reid_min_obs_without_embedding`
  observaciones para re-identificar (una detección espuria aislada no toma una identidad oculta). Un cambio de
  track sin hueco (la identidad se observó en el frame anterior) se reporta `TRACKED`, no `REIDENTIFIED`.
- RF-4c (revisión S3): si el equipo observado contradice al de la identidad del tracklet, esa caja no se emite ni
  actualiza la identidad en ese frame. Un `track_id` repetido en el mismo frame: se conserva la observación de
  mayor confianza.
- RF-5: Partición de tracklets: si la apariencia de un tracklet se aleja de su referencia por encima de
  `split_distance` durante `split_frames` frames consecutivos, o su equipo cambia de forma sostenida, el tracklet
  se parte y el fragmento nuevo pasa por RF-3/RF-4.
- RF-6: Durante una superposición (IoU entre cajas de identidades vigentes > `overlap_iou`), no se actualiza la
  galería de apariencia de esas identidades.
- RF-6b: Resolución de la separación. El grupo es la componente conexa de identidades que se superpusieron entre
  sí (en cadena). En cada frame, los tracklets **ya libres** del grupo se reasignan por apariencia (galería previa a
  la superposición) y equipo entre las identidades del grupo en juego; si un libre toma la identidad de un tracklet
  aún superpuesto, ese recibe la que el libre dejó. Se aplica si mejora el costo en más de `swap_margin`. El grupo
  vence `separation_frames` frames después de que ningún miembro está superpuesto. Al reasignar, solo se
  re-atribuyen las lecturas de dorsal tomadas desde el inicio de la superposición. (Revisión S3: la versión
  anterior, con ventana por identidad, dejaba intercambios persistentes en cadenas de 3 jugadores.)
- RF-7: Dorsal por votación a nivel identidad: se asigna solo con ≥ `jersey_min_reads` lecturas, confianza media
  ≥ `jersey_min_conf` y ≥ `jersey_min_share` del voto total. Unicidad por equipo (no global). Si dos identidades del
  mismo equipo reclaman un número, lo conserva la de mayor evidencia y la otra queda en `null`.
- RF-8: Una identidad nueva solo se crea si ninguna identidad no vigente del equipo es compatible y el cupo de
  identidades vigentes del equipo (`TRACKED`/`OCCLUDED`) es < `roster_size` (6). Sin cupo, el tracklet queda sin
  identidad (no se duplica jugadores).
- RF-9: Sustitución: una identidad `LOST` no se reasigna a un tracklet incompatible en apariencia aunque haya cupo;
  el entrante recibe identidad nueva.

## Requerimientos no funcionales
- RNF-1: Determinista: misma entrada → misma salida (sin aleatoriedad).
- RNF-2: ≤ 2 ms por frame con 20 observaciones en CPU (no bloquea el objetivo de ~15 FPS del pipeline).
- RNF-3: Sin dependencias nuevas (numpy, scipy, pydantic ya presentes).

## Casos de uso
- CU-1: El pipeline (S4) llama `update(frame_idx, observations)` por frame y serializa la salida a JSONL.
- CU-2: Los tests de escenarios alimentan secuencias sintéticas con ground truth conocido.

## Reglas de negocio
- RN-1: `PLAYER ID ≠ TRACK ID`; ningún track nuevo implica jugador nuevo.
- RN-2: Ninguna señal sola define la identidad; el equipo es condición necesaria, no suficiente.
- RN-3: Ante duda, el dorsal es `null` (`jersey_confidence = 0`).
- RN-4: El lado de la cancha no define el equipo (el equipo es una entrada del clasificador).

## Estados del jugador

| Estado | Significado |
|---|---|
| `DETECTED` | Frame en que se crea la identidad (su tracklet alcanzó `confirm_frames` observaciones) |
| `TRACKED` | Observada en este frame con su tracklet vigente |
| `OCCLUDED` | Sin observación hace ≤ `lost_after` frames |
| `LOST` | Sin observación hace > `lost_after` frames; sigue disponible para Re-ID |
| `REIDENTIFIED` | Frame en que un tracklet nuevo se asoció a una identidad OCCLUDED/LOST |

Transiciones: `DETECTED → TRACKED` (tras confirmación); `TRACKED → OCCLUDED` (sin observación);
`OCCLUDED → TRACKED` (vuelve el mismo tracklet); `OCCLUDED → LOST` (> `lost_after`);
`OCCLUDED|LOST → REIDENTIFIED` (asociación de otro tracklet) `→ TRACKED` (frame siguiente).
Las identidades `OCCLUDED`/`LOST` no aparecen en la salida del frame (no hay caja), pero conservan su ID.

## Criterios de aceptación
- [ ] AC-1: dado un jugador visible 300 frames con el mismo track, cuando se procesa, entonces aparece en la salida desde el frame `confirm_frames − 1` (estado `DETECTED`), tiene un único `player_id` en todos los frames emitidos y estado `TRACKED` desde el frame `confirm_frames`.
- [ ] AC-2: dado un jugador que desaparece 30 frames y reaparece con otro track cerca de su trayectoria, cuando se procesa, entonces conserva el mismo `player_id` y el primer frame de reaparición tiene estado `REIDENTIFIED`.
- [ ] AC-3: dado dos jugadores del mismo equipo que se cruzan y cuyo tracker intercambia los tracks en el cruce, cuando se separan, entonces a más tardar `separation_frames` frames después de la separación cada `player_id` sigue al mismo jugador real que antes del cruce (0 intercambios persistentes) y nunca hay `player_id` duplicados en un frame.
- [ ] AC-4: dado lecturas de dorsal "7" con confianza ≥ 0,9 en ≥ `jersey_min_reads` frames, cuando se procesan, entonces la identidad tiene `jersey_number = 7` y `jersey_confidence ≥ 0,9`.
- [ ] AC-5: dado un jugador sin lecturas de dorsal, cuando se procesa, entonces `jersey_number = null`, `jersey_confidence = 0` y su `player_id` es estable.
- [ ] AC-6: dado un jugador sin lecturas durante 200 frames y con lecturas "8" después, cuando se procesa, entonces conserva el mismo `player_id` y termina con `jersey_number = 8`.
- [ ] AC-7: dado un fallo del detector de 5 frames sin cambio de track, cuando se procesa, entonces la identidad pasa a `OCCLUDED` en el hueco y vuelve a `TRACKED` con el mismo `player_id`.
- [ ] AC-8: dado un tracklet espurio de 2 frames, cuando se procesa, entonces no se crea ninguna identidad.
- [ ] AC-9: dado un jugador con apariencia distinta a todos los existentes que entra con cupo disponible, cuando se confirma, entonces recibe una identidad nueva.
- [ ] AC-10: dado un jugador que sale (LOST) y otro de apariencia distinta que entra por el mismo lugar, cuando se procesa, entonces el entrante recibe una identidad nueva y la del saliente no se reutiliza.
- [ ] AC-11: dado 6 + 6 jugadores con cruces, equipo A con lecturas de dorsal y equipo B sin lecturas, cuando se procesa el segmento, entonces hay exactamente 12 `player_id` distintos, cada jugador real mapea a uno solo fuera de las ventanas de superposición (misma tolerancia que AC-3; errores transitorios < 5 % de los frames), B tiene todos sus dorsales en `null`, y si luego aparecen lecturas "9" para un jugador de B su identidad toma `jersey_number = 9` sin cambiar de `player_id`.
- [ ] AC-12: dado una única lectura "1" con confianza 0,6, cuando se procesa, entonces el dorsal sigue en `null`.
- [ ] AC-13: dado dos identidades del mismo equipo con lecturas "5", cuando se procesan, entonces solo la de mayor evidencia tiene `jersey_number = 5`; y dado el mismo número en equipos distintos, entonces ambos lo conservan.
- [ ] AC-14: dado el equipo A con 6 identidades vigentes, cuando aparece un séptimo tracklet incompatible del equipo A, entonces no se crea una séptima identidad vigente.
- [ ] AC-15: dado cualquier frame procesado, cuando se serializa, entonces la salida valida contra el esquema `FrameOutput` (campos del punto 13 del MVP: `frame`, `players[]` con `player_id`, `track_id`, `team_id`, `jersey_number`, `jersey_confidence`, `bbox`, `confidence`, `state`).
- [ ] AC-16: dado la misma secuencia de entrada dos veces, cuando se procesa, entonces las salidas son idénticas.
- [ ] AC-17: dado un cruce entre jugadores de equipos distintos con intercambio de tracks del tracker, cuando se separan, entonces ningún `player_id` cambia de equipo.

## Limitación conocida (medida)
Si el tracker intercambia IDs **al empezar** una superposición, la salida muestra el intercambio mientras dure la
superposición y se corrige al separarse (RF-6b); durante la superposición las apariencias están contaminadas y no se
usan a propósito. Medición (corregida tras la revisión independiente, que mostró que la primera medición se había
hecho con una sola semilla): en el escenario AC-11 con 15 combinaciones de disposición × semilla
(`test_finding1_ac11_holds_across_seeds`), 12 identidades exactas, 0 errores fuera de las ventanas de
superposición y < 5 % de frames con error transitorio en todas. Mejora posible (fuera de alcance): detectar la
inversión brusca de velocidad del tracklet durante la superposición.

## Restricciones técnicas
Python 3.12, numpy, scipy (`linear_sum_assignment`), pydantic. Paquete `volley_cv.identity`. Tipado estricto (mypy).

## Manejo de errores
- Observación con caja inválida (ancho/alto ≤ 0) o embedding con norma 0: se descarta y se registra.
- Embedding ausente: la asociación usa solo movimiento + dorsal, con umbral más estricto.
- Frames no consecutivos (salto hacia atrás): `ValueError` (el llamador debe resetear con `reset()` en un corte).
- Equipo desconocido en la observación: el tracklet espera (no crea identidad) hasta tener equipo.

## Seguridad
Sin impacto: no hay entradas externas no confiables ni datos personales (los IDs son artificiales). Los embeddings
son vectores numéricos sin información identificable fuera del sistema.

## Estrategia de testing
| AC | Test |
|---|---|
| AC-1…AC-14, AC-16, AC-17 | Escenarios sintéticos deterministas en `tests/scenarios/` (generador `synth.py` con ground truth, embeddings por jugador con componente de equipo, simulación de cortes e intercambios del tracker) |
| AC-15 | Unit test del esquema (`tests/unit/test_output_schema.py`) |
| Partición, votación, máquina de estados | Unit tests de cada componente |
| Evaluación real | S6 con clips anotados (IDF1, ID switches) |

## Dependencias e impacto
Nuevo paquete `volley_cv.identity` y `volley_cv.output`. Sin dependencias nuevas. Consumido por el pipeline de S4.
