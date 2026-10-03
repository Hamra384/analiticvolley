# Informe de sprint S5b — Pelota: tracker con estados y detector propio

- **Fecha:** 2026-10-03 · **Issue:** #21 · **Rama:** `21-s5b-ball` · **Spec:** SPEC-005 · **Spike:** SPIKE-002
- **Origen:** tracking de pelota del MVP 1. SPIKE-002 había medido que YOLO COCO es preciso (0,92) pero ve la pelota
  en ~1 de cada 3 frames, y recomendaba entrenar un detector específico.

## Qué se hizo

1. **BallTracker** (Kalman de aceleración constante) con estados `DETECTED` / `TRACKED` / `PREDICTED` / `LOST` /
   `REACQUIRED`. Una predicción nunca sale como `DETECTED`/`TRACKED` (regla de producto). Predicción de hasta 8
   frames; después `LOST` sin posición.
2. **Integración**: `ball` en cada frame del JSONL; en el video, círculo lleno = detectada, contorno = predicha.
3. **Dataset con pseudo-etiquetas** (`tools/ball_dataset.py`): 231 ventanas de 2 s de los dos partidos, todas a más de
   60 s de cualquier clip de evaluación (test con el catálogo real). 1738 frames etiquetados; revisión visual de 24
   al azar: 23 sobre la pelota, 1 dudosa.
4. **Detector propio** (`tools/train_ball.py`): YOLOv8s, 1 clase, imgsz 1280, 36 épocas (la tarea se cortó por el
   límite de 2 h; se usó el mejor punto, época 34, que ya no mejoraba). Pesos fuera del repo
   (`<datos>/models/ball_v1.pt`, derivan de transmisiones con derechos de terceros).
5. **Filtro de señuelos** (RF-2b): una pelota quieta más de 1,5 s se ignora.

## Resultados (métricas proxy de SPIKE-002, mismo tracker para ambos detectores, 10 clips)

| | COCO (YOLOv8m clase 32) | Propio (ball_v1) |
|---|---|---|
| P1 cobertura mediana (mín) | 0,345 (0,193) | **0,563 (0,492)** |
| Clips en que es mejor | — | **10 / 10** |
| P4 precisión visual | 0,92 (SPIKE-002) | **≈ 0,96** (46/48 sobre la pelota del juego en A2, A3, A5, K4) |
| P5 velocidad del detector | 17–18 FPS | **32–37 FPS** |

- **Hallazgo que cambió el resultado:** la primera medición del detector propio daba 0,63, pero en A2, 203 de 378
  frames "con pelota" eran una **pelota de repuesto** en la mano de un alcanzapelotas (se veía en la hoja de
  revisión, no en la métrica). Con el filtro de señuelos la cobertura real quedó en 0,56.
- **Decisión (del usuario):** el criterio fijado antes de entrenar exigía P1 ≥ 0,6. El detector propio quedó en
  0,56 pero supera al COCO en todo: se adopta como **excepción documentada** (SPEC-005 RF-8b). El recall real
  (contra anotación humana) se mide en S6.
- **Revisión visual del usuario:** aprobada ("la veo perfecto").

## Revisión independiente (subagente reviewer)

Sin caminos en que una predicción salga como detectada. 2 medios y 6 bajos, corregidos con test de regresión:
- **M1** tras `LOST`, un candidato de muy baja confianza lejano salía `REACQUIRED` (la predicción seguía extrapolando
  sin límite) → en `LOST` la posición se congela, el radio no crece y `REACQUIRED` exige confianza y plazo.
- **M2** el filtro de señuelos marcaba como quieta una pelota lenta pero en movimiento → ancla fija.
- Bajos: doble conteo de señuelos, costo con muchos candidatos, caché por objeto frame (no `id()`), pasada de
  personas sin pedir la clase pelota cuando no hace falta, spec RF-7 alineada con el código, guarda de FPS inválido,
  test de punta a punta del camino por defecto de la CLI.

## Pruebas

- 378 tests locales (0 salteados); ruff, format y mypy limpios. CI: ver PR.

## Velocidad de punta a punta

El detector propio es una segunda pasada (además de YOLOv8m para personas). Corrida final de 6 clips:
A2/A3/A4 4,0–4,4 FPS (antes ~5,0); K2/K3/K5 6,0–6,6 FPS (antes ~8,5). Costo 15–25 %, dentro de RNF-3 (≤ 30 %).
Sigue lejos de la meta de 15 FPS (#16).

Estados de la pelota en esa corrida (600 frames por clip): detectada/seguida 292–474, predicha 47–170, perdida
46–236. Jugadores sin cambios respecto de S5c (13–14 IDs por clip, 0 `player_id` duplicados).

## Errores míos en este sprint

- Medí la cobertura sin mirar qué se estaba siguiendo: la pelota de repuesto inflaba la métrica. Lo detecté en la
  revisión visual antes de decidir.
- Dos fallas de entrenamiento antes de arrancar (opción `deterministic` y memoria de la placa).
- Para detener una corrida usé un comando que cierra **todos** los procesos Python de la máquina; se lo avisé al
  usuario. Desde ahora se detiene solo la tarea propia.

## Riesgos y pendientes

- La pelota en la mano del sacador antes del saque puede salir `LOST` (costo del filtro de señuelos).
- Repeticiones de la transmisión de un rally de evaluación emitidas lejos del original podrían haber entrado al
  entrenamiento (riesgo bajo, no verificado).
- Las predicciones de 8 frames pueden caer sobre el piso cuando la pelota sale de cuadro; se marcan como predichas.
- **S6 bloqueada:** necesita la corrección de las pre-anotaciones en CVAT (usuario).

## Próximo

S6: evaluación con ground truth (IDF1, intercambios de ID, precisión del número, recall de pelota) en cuanto estén
las anotaciones. Mientras tanto, backlog #13–#16.
