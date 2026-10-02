# Informe de sprint S5a — Número de camiseta como señal de identidad

- **Fecha:** 2026-10-02 · **Issue:** #17 · **Rama:** `17-s5a-jersey` · **Spec:** SPEC-003 · **Spike:** SPIKE-004
- **Origen:** pedido del usuario: vincular la identidad con el número de camiseta para reconocer al jugador cuando
  vuelve. Ataca los issues #13 (líberos en el equipo contrario) y #14 (identidades que mezclan o fragmentan).

## Qué se hizo

1. **SPIKE-004**: easyocr (imagen directa, solo dígitos, confianza ≥ 0,95) contra easyocr con CLAHE y contra el
   clasificador legado. Ganó easyocr directo: **precisión 0,96, cobertura 0,30 por recorte** (de cada 10 torsos con
   número visible, lee 3 y casi nunca inventa).
2. **Lector en el pipeline**: solo cajas grandes (≥ 18 % del alto), cada 5 frames por track, y **no** cuando otra
   persona más cerca de la cámara tapa el torso.
3. **Identidad**: votación del número por identidad (único por equipo), **fusión** de identidades del mismo equipo
   con el mismo número que nunca estuvieron en el mismo frame, **partición** de un tracklet cuando el número cambia,
   `merges.json` y reescritura del JSONL con la identidad final.
4. **Líbero por número**: `libero_numbers` por equipo en la config de cada video. El número decide el equipo solo
   cuando el color del torso es compatible con ese líbero; se recuerda por track hasta un corte o una superposición.

## Resultados en video real (revisión visual por identidad, sin ground truth)

| Clip | Números asignados (comprobados a ojo en la hoja) | Efecto visible |
|---|---|---|
| A2 | Japón #25, #7, #28, #35; Argentina #24 | Correctos. Un argentino del fondo había tomado el #35 de un japonés que lo tapaba: corregido (regla de oclusión) |
| A3 | Japón #18, #28; Argentina #19 (líbero), #6, #1, #7, #8 | El líbero #19 estaba como **Japón 387 frames**; ahora es Argentina (B01) en 480 frames y ninguna identidad japonesa lleva el 19 |
| K2 | Japón #9, #8, #5 | El #5 es un japonés real (rojo), confirmado a ojo. Una versión intermedia lo mandaba a Corea (el líbero coreano también es el #5): corregido |
| K5 | Japón #9; Corea #30 | — |

- **Fusiones por número: 0 en los 4 clips.** Hecho: ninguna identidad provisoria llegó a 3 lecturas confiables del
  mismo número que otra. Hipótesis: con cobertura 0,30, solo cajas grandes y lectura cada 5 frames, los fragmentos
  cortos (la mayoría < 50 frames) no juntan evidencia antes de terminar. El mecanismo está probado con escenarios,
  pero en video real **todavía no redujo la fragmentación** (#14 sigue abierto).
- Sin `player_id` duplicados en ningún frame de los 4 clips (verificado sobre el JSONL).
- Velocidad: A2 5,1 FPS, A3 5,1, K2 8,6, K5 8,5 (antes de S5a: A2 5,8, K2 9,7). Costo del OCR ≈ 10–12 %; entre
  corridas del mismo clip varió entre 4,4 y 5,4 FPS en A2, así que la cifra es aproximada.

### Líbero y rival con el mismo número (pedido del usuario, sin plantel)

Si el color del líbero de un equipo es igual al principal del otro (blanco en JPN-ARG) y un rival lleva el número
del líbero, el número solo no alcanza. RF-6d: el número no decide si la persona está **del lado del rival** según la
evidencia de lado del tramo; sin evidencia (plano lateral) decide el número. Re-corrida de los 4 clips: A2, A3 y K5
idénticos (el #19 de A3 sigue en Argentina, 480 frames); en K2, B05 queda coreano todo el tramo y A09 (#5 de Japón)
sube de 246 a 282 frames. Límite: cerca de la red la posición es menos confiable.

## Revisión independiente (subagente reviewer)

1 hallazgo alto y 4 medios. Todos se corrigieron con un test de regresión, salvo M4, que quedó documentado (tests
en rojo observados antes de cada corrección):
- **H1** tras una fusión, dos tracklets podían quedar vinculados al mismo jugador → `player_id` duplicado en un frame
  (reproducido). Ahora queda vinculado un solo tracklet, el más reciente.
- **M1** una fusión en el frame de creación dejaba la identidad en `DETECTED` para siempre → `REIDENTIFIED` y luego
  `TRACKED`.
- **M2** la fusión movía lecturas atribuidas a una tercera identidad → solo mueve las de la identidad fusionada.
- **M3** el número recordado podía arrastrarse si el tracker pasaba el ID a otra persona → se olvida en una
  superposición (IoU > 0,3).
- **M4** (configuración) con colores idénticos (blanco de Japón = líbero argentino) el número de líbero decide solo:
  si Japón tuviera un #19, se iría a Argentina. Documentado en la config; el usuario confirmó que Japón no tiene #19 en este partido.
- Bajos: socios de superposición simétricos tras fusión (corregido), test de AC-5 más estricto (corregido).

## Pruebas

- 324 tests locales (0 salteados); ruff, format y mypy limpios; cobertura 98,3 % (base 98,4 %, dentro de la
  tolerancia). CI: ver PR.
- Criterios agregados durante el sprint por evidencia del video: RF-2b/AC-10 (oclusión), RF-6b/AC-11 (número
  recordado), RF-6c/AC-12 (color compatible), RF-6d/AC-14 (lado del rival), AC-13 (revisión).

## Errores míos en este sprint

- La primera versión del "número recordado" (RF-6b) empeoró K2: lo detecté en la revisión visual y lo corregí antes
  del PR. Un cambio que mejora un clip puede romper otro: por eso se revisan los 4.
- Un test asumía 4 tracks y el escenario tenía un quinto: corregí el test, no el código.

## Riesgos y pendientes

- El número todavía no fusiona fragmentos en video real (0 fusiones). Opciones a evaluar: leer más seguido cuando el
  track es nuevo, bajar el mínimo de lecturas para fusionar solo si no hay conflicto, o un lector con más cobertura.
  Sin anotaciones no se puede medir el costo en precisión.
- Siguen: identidades que mezclan personas en la red (#14), oficiales de traje y gráficos (#15), velocidad (#16).
- Líbero y rival con el mismo número y el mismo color: cubierto por posición (RF-6d) salvo cerca de la red o sin evidencia de lado. En JPN-ARG no ocurre (verificado por el usuario).

## Próximo

S5b: pelota (detector específico y estados DETECTED/PREDICTED/LOST). Para S6 hace falta la anotación en CVAT.
