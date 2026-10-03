# SPEC-005: Pelota — tracker con estados y detector específico

- **Issue:** #21
- **Estado:** aprobada (plan del MVP 1; orden S4.1 → S5a → S5b acordado con el usuario)
- **ADR relacionados:** ninguno nuevo (dentro de ADR-0002). Los pesos entrenados quedan **fuera del repo** (derivan
  de transmisiones con derechos de terceros), en el directorio de datos, referenciados por config.
- **Spike:** SPIKE-002 (`docs/spikes/002-pelota.md`): YOLOv8m COCO es preciso (P4 0,92) pero cubre poco (P1 0,36);
  recomienda un detector específico entrenado con pseudo-etiquetas.

## Problema
El MVP 1 pide la posición de la pelota por frame con su estado. Ningún detector evaluado llega a la cobertura
necesaria, y aun con un buen detector va a haber frames sin detección (pelota tapada, borrosa o fuera de cuadro).

## Objetivos y alcance
- Dentro: `BallTracker` con estados; integración al pipeline (salida `ball` y video de debug); detector específico
  entrenado con pseudo-etiquetas fuera de los clips de evaluación; métricas proxy de SPIKE-002 para decidir.
- Fuera: altura 3D, toques, jugadas, posesión; evaluación con ground truth (S6).

## Requerimientos funcionales
- RF-1: `BallDetector` (interfaz) devuelve candidatos `(cx, cy, w, h, conf)` por frame. Implementación YOLO con pesos
  configurables (COCO clase 32 o un modelo propio de 1 clase).
- RF-2: `BallTracker` con filtro de Kalman de aceleración constante en píxeles. Por frame elige a lo sumo un
  candidato: el más cercano a la predicción dentro de un radio que crece con los frames sin detección; sin track,
  el de mayor confianza ≥ `start_conf`.
- RF-2b (agregado al medir: en A2, 203 de 378 frames "con pelota" eran una pelota de repuesto en la mano de un
  alcanzapelotas): una detección que permanece quieta (radio ≈ 0,6 diámetros) durante `static_frames` (1,5 s) es
  un señuelo y se ignora mientras siga quieta. Costo: la pelota en la mano del sacador antes del saque puede salir
  `LOST`.
- RF-3: estados por frame:
  - `DETECTED`: primera detección de un track nuevo.
  - `TRACKED`: detección asociada al track existente.
  - `PREDICTED`: sin detección; posición predicha, como máximo `max_predict` frames seguidos; confianza decreciente.
  - `LOST`: más de `max_predict` frames sin detección; `position = null`.
  - `REACQUIRED`: primera detección después de `LOST`, compatible con la última trayectoria.
- RF-4: una posición predicha **nunca** se reporta como `DETECTED`/`TRACKED` (regla de producto).
- RF-5: corte de escena → el track se reinicia (`LOST` hasta una nueva detección, que es `DETECTED`).
- RF-6: el pipeline agrega `ball` a cada `FrameOutput` y la dibuja en el video de debug (marca distinta para
  detectada y predicha).
- RF-7: **detector específico.** Pseudo-etiquetas de YOLOv8m COCO (conf ≥ 0,25) enlazadas por el tracker; se
  aceptan detecciones dentro de trayectorias de ≥ 5 frames y se interpolan huecos de ≤ 3 frames entre detecciones.
  Solo tramos a más de 60 s de cualquier clip de evaluación. Solo se usan como imágenes de entrenamiento los frames
  con etiqueta (un frame sin etiqueta puede tener pelota no detectada). Fine-tuning de YOLOv8 (1 clase,
  imgsz 1280).
- RF-8: **criterio de adopción** (fijado antes de entrenar, mismas métricas que SPIKE-002 sobre los 10 clips de
  evaluación): el detector propio reemplaza al COCO si su P1 (cobertura del track) mediana es ≥ 0,6 **y** mayor que
  la del COCO, con P4 (precisión visual, 12 recortes al azar por clip en al menos 4 clips) ≥ 0,8. Si no, se queda el
  COCO y se documenta.

## Requerimientos no funcionales
- RNF-1: el tracker es determinista y corre en CPU en < 1 ms por frame.
- RNF-2: sin dependencias nuevas (ultralytics ya está en el extra `ml`).
- RNF-3: el detector de pelota no debe bajar el FPS de punta a punta más de un 30 % (se reporta).

## Casos de uso
- CU-1: la pelota se pierde 5 frames tras un remate (borrosa): esos frames salen `PREDICTED` con posición; al
  reaparecer, `TRACKED`.

## Reglas de negocio
- RN-1: una predicción nunca es `DETECTED` (CLAUDE.md, reglas de producto).
- RN-2: ante duda, `LOST` sin posición antes que una posición inventada lejos de la evidencia.

## Criterios de aceptación
- [ ] AC-1: dado una secuencia de detecciones sobre una parábola, cuando se procesa, entonces el primer frame es `DETECTED`, los siguientes `TRACKED` y la posición sigue a la detección.
- [ ] AC-2: dado un hueco de k ≤ `max_predict` frames en la parábola, cuando se procesa, entonces esos frames son `PREDICTED` con posición a menos de 2 diámetros de la verdadera y la siguiente detección es `TRACKED`.
- [ ] AC-3: dado un hueco mayor que `max_predict`, cuando se procesa, entonces después de `max_predict` frames el estado es `LOST` con `position = null`, y la siguiente detección compatible es `REACQUIRED`.
- [ ] AC-4: dado un frame sin detección, cuando se procesa, entonces el estado nunca es `DETECTED` ni `TRACKED`.
- [ ] AC-5: dado un falso positivo lejano mientras hay track, cuando se procesa, entonces no se asocia (se elige el candidato compatible o se predice).
- [ ] AC-6: dado un corte de escena, cuando se reinicia, entonces el estado es `LOST` hasta la próxima detección, que es `DETECTED`.
- [ ] AC-10: dado un señuelo quieto detectado siempre y la pelota del juego en vuelo, cuando se procesa, entonces el track nunca queda en el señuelo; un señuelo solo termina en `LOST`; una pelota lenta pero en movimiento no es señuelo.
- [ ] AC-7: dado el pipeline con un detector de pelota falso, cuando corre, entonces cada frame del JSONL tiene `ball` con estado válido y el video de debug se escribe.
- [ ] AC-8: dado los segmentos de entrenamiento, cuando se generan, entonces ningún frame cae a menos de 60 s de un clip de evaluación (test del generador con el catálogo real).
- [ ] AC-9: dado el detector entrenado, cuando se mide con las métricas de SPIKE-002 en los 10 clips, entonces se aplica RF-8 y la decisión queda documentada en el informe (evidencia manual de P4).

## Restricciones técnicas
GTX 1660 Super 6 GB: YOLOv8s a imgsz 1280 con batch chico. Entrenamiento en segundo plano con CVAT pausado.

## Manejo de errores
- Detector que falla en un frame: sin candidatos (el tracker predice).
- Pesos propios no encontrados: error de configuración claro (no se cae a COCO en silencio).

## Seguridad
Los frames de entrenamiento y los pesos derivados quedan fuera del repo público (derechos de terceros);
`check_no_media` lo controla.

## Estrategia de testing
| AC | Test |
|---|---|
| AC-1…AC-6, AC-10 | unit del tracker con trayectorias sintéticas (`tests/unit/test_ball_tracker.py`) |
| AC-7 | integración del pipeline con detector falso (`tests/integration/test_ball_pipeline.py`) |
| AC-8 | unit del generador de segmentos (`tests/unit/test_ball_segments.py`) |
| AC-9 | medición proxy + revisión visual manual |

## Dependencias e impacto
Nuevo `volley_cv.ball` (tracker, interfaz de detector, adaptador YOLO). Modifica `pipeline`, `viz`, `__main__`
(flag de pesos de pelota) y la config de datos. Nuevo `tools/ball_dataset.py` y `tools/train_ball.py`.
