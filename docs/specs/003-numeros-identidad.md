# SPEC-003: Número de camiseta como señal de identidad

- **Issue:** #17
- **Estado:** aprobada (pedido del usuario, 2026-10-02)
- **ADR relacionados:** ninguno nuevo (extiende SPEC-001 dentro de ADR-0002)
- **Spike:** SPIKE-004 (`docs/spikes/004-numeros.md`): easyocr con confianza ≥ 0,95

## Problema
Tras S4.1, los errores restantes más visibles son líberos asignados al equipo contrario (#13) e identidades que
mezclan o fragmentan jugadores (#14). La apariencia por color no distingue bien a compañeros con la misma camiseta,
y la posición en la imagen no alcanza para el líbero. El número de camiseta es único por equipo en un partido.

## Objetivos y alcance
- Dentro: leer el número en el pipeline; usar el número para (a) reconocer al jugador que vuelve, (b) fusionar una
  identidad provisoria con la original, (c) partir un tracklet cuando el número cambia, (d) resolver el equipo del
  líbero con los números de líbero configurados por video; reescritura de la salida con las fusiones.
- Fuera: entrenar un lector propio; leer números de jugadores lejanos; nombres en la camiseta.

## Requerimientos funcionales
- RF-1: `JerseyReader` (interfaz) devuelve `JerseyRead(número, confianza)` o nada para una caja. La implementación
  easyocr emite solo lecturas con confianza ≥ `min_conf` (0,95) y 1–2 dígitos.
- RF-2: el pipeline lee el número solo de cajas con altura ≥ `min_height` (18 % del frame) y cada `stride` frames
  por track (costo acotado).
- RF-2b (agregado en la revisión visual de A2): no se lee una caja cuyo torso esté tapado más de
  `max_occlusion` (15 %) por otra caja con los pies más abajo (más cerca de la cámara). Evidencia: en A2 un
  argentino del fondo heredó el #35 de un japonés en primer plano que le tapaba la caja.
- RF-3: **fusión por número.** Cuando una identidad X toma el número N (RF-7 de SPEC-001) y otra identidad Y del
  mismo equipo ya tiene N, y X e Y **nunca se observaron en el mismo frame**, son el mismo jugador: X se fusiona en
  la más antigua (Y). El tracklet de X pasa a Y, X deja de existir, y la fusión (X → Y, frame) queda registrada.
  Si coexistieron, es un conflicto y se mantiene RF-7 (gana la de mayor evidencia, la otra queda en `null`).
- RF-4: **reescritura.** Al terminar, el pipeline aplica las fusiones a todo el JSONL (las apariciones previas de X
  pasan a ser Y) y guarda `merges.json`. Como X e Y nunca coexistieron, no se producen duplicados.
- RF-5: **partición por número.** Si un tracklet con número establecido (≥ `jersey_min_reads` lecturas de N)
  acumula `jersey_split_reads` lecturas consecutivas de otro número M, se parte (el tracker cambió de persona).
- RF-6: **líbero por número.** La config del video puede listar `libero_numbers` por equipo. Si el color del torso
  es compatible con el prototipo de líbero de un equipo Y y la lectura es un número de líbero de Y, la observación
  es de Y (prevalece sobre la prior por cantidad de RF-3d de SPEC-002).
- RF-6b (agregado en la revisión visual de A3): para RF-6 vale el último número leído del track hasta un corte,
  no solo el del frame de la lectura. Evidencia: el líbero #19 de Argentina se leía en pocos frames y entre
  lecturas quedaba en Japón (A05, 387 frames). El número recordado se olvida cuando la caja se superpone con otra
  (IoU > 0,3), donde el tracker puede pasar el ID a otra persona (revisión S5a M3).
- RF-6c (agregado en la revisión visual de K2): RF-6 aplica solo si el torso está al menos tan cerca del líbero
  dueño del número como del color principal del rival (tolerancia 2). Evidencia: Japón tiene un #5 real (rojo) y
  el líbero coreano es el #5 (azul marino); con RF-6b un rojo en sombra quedaba en Corea.
- RF-6d (pedido del usuario, sin plantel): RF-6 tampoco aplica si la persona está del lado del rival: con evidencia
  de lado de ambos equipos (separada al menos `min_separation`), los pies más cerca de la mediana del rival que de
  la del dueño del número. Cubre colores idénticos (blanco de Japón = líbero argentino) con un rival que lleve el
  número del líbero. Sin evidencia de lado (plano lateral) decide el número. Límite conocido: cerca de la red la
  posición es menos confiable.

## Requerimientos no funcionales
- RNF-1: la lectura no debe bajar el FPS de punta a punta más de un 40 % (se reporta).
- RNF-2: sin dependencias nuevas fuera de easyocr (extra `ml`, Apache-2.0).

## Casos de uso
- CU-1: un jugador sale de cuadro, vuelve con otra identidad provisoria y, al leerse su número, la provisoria se
  fusiona en la original; en el JSONL final todo el recorrido tiene la identidad original.

## Reglas de negocio
- RN-1: el número es único por equipo dentro de un video.
- RN-2: ante duda, sin número (`null`); una fusión requiere el número confirmado por votación (RF-7 de SPEC-001).

## Criterios de aceptación
- [ ] AC-1: dado dos identidades del mismo equipo que nunca coexistieron y que confirman el mismo número, cuando se procesa, entonces la más nueva se fusiona en la más antigua, el tracklet sigue con el `player_id` antiguo y la fusión queda registrada.
- [ ] AC-2: dado dos identidades del mismo equipo que **coexistieron** y confirman el mismo número, cuando se procesa, entonces no se fusionan (se mantiene RF-7: la de menor evidencia queda en `null`).
- [ ] AC-3: dado el mismo número en equipos distintos, cuando se procesa, entonces no se fusionan.
- [ ] AC-4: dado un JSONL con frames de X antes de la fusión X → Y, cuando se aplica la reescritura, entonces todas las apariciones de X pasan a Y y ningún frame tiene `player_id` duplicados.
- [ ] AC-5: dado un tracklet con número 7 establecido que pasa a leer 11 de forma consecutiva, cuando acumula `jersey_split_reads` lecturas, entonces el tracklet se parte y el número 11 no se suma a la identidad del 7.
- [ ] AC-6: dado un torso blanco ambiguo (principal de A = líbero de B) con lectura de un número de líbero de B, cuando se clasifica, entonces el equipo es B; y sin lectura, la prior (A) se mantiene.
- [ ] AC-7: dado una caja más chica que `min_height` o un track leído hace menos de `stride` frames, cuando corre el pipeline, entonces no se llama al lector.
- [ ] AC-10: dado una caja cuyo torso está tapado por otra con los pies más abajo, cuando corre el pipeline, entonces no se lee esa caja pero sí la de adelante; con un roce menor a `max_occlusion` se leen ambas.
- [ ] AC-11: dado un track con un número leído una vez, cuando siguen los frames sin lectura, entonces el clasificador de equipo recibe ese número en cada frame hasta un corte.
- [ ] AC-12: dado un torso ambiguo pero más cerca del principal del rival que del líbero dueño del número leído, cuando se clasifica, entonces el número no cambia el equipo.
- [ ] AC-13 (revisión S5a): dado una fusión X → Y, cuando el tracker revive el tracklet viejo de Y junto al de X, entonces ningún frame tiene `player_id` duplicado (un solo tracklet vinculado por identidad) y una fusión en el frame de creación se reporta `REIDENTIFIED`, no `DETECTED`.
- [ ] AC-14: dado un torso del color ambiguo con el número del líbero de Y, cuando la persona está del lado del rival de Y, entonces el número no decide (queda en el equipo del rival); del lado de Y, es de Y.
- [ ] AC-8: dado una lectura con confianza < `min_conf` o con más de 2 dígitos, cuando la procesa el lector easyocr, entonces no se emite (test con OCR falso).
- [ ] AC-9: dado los clips A2, A3, K2 y K5 reales, cuando corre el pipeline, entonces la revisión visual muestra menos líberos en el equipo contrario e identidades fusionadas por número (evidencia manual en el informe).

## Restricciones técnicas
easyocr importado de forma diferida (extra `ml`). Tests con lectores falsos (CI sin GPU).

## Manejo de errores
- El lector falla o devuelve texto no numérico: no hay lectura (la observación sigue sin número).
- Recorte vacío: no hay lectura.
- Una fusión cuyo destino ya no existe (fusiones encadenadas X → Y → Z): se resuelve al destino final.

## Seguridad
Sin impacto: no hay datos personales (los números son públicos en la transmisión y no se asocian a nombres).

## Estrategia de testing
| AC | Test |
|---|---|
| AC-1…AC-3, AC-5, AC-13 | escenarios de identidad (`tests/scenarios/test_jersey_identity.py`) |
| AC-4 | unit de reescritura (`tests/unit/test_s5a_units.py`) e integración (`tests/integration/test_jersey_pipeline.py`) |
| AC-6, AC-12, AC-14 | unit de equipo (`tests/unit/test_s5a_units.py`) |
| AC-7, AC-10, AC-11 | integración del pipeline con lector falso (`tests/integration/test_jersey_pipeline.py`) |
| AC-8 | unit del lector con OCR falso (`tests/unit/test_s5a_units.py`) |
| AC-9 | revisión visual manual |

## Dependencias e impacto
Modifica `volley_cv.identity.manager` (fusión, partición por número), `volley_cv.pipeline` (lectura y
reescritura), `volley_cv.team` (líbero por número), `video_config` (`libero_numbers`). Nuevo
`volley_cv.jersey_reader`.
