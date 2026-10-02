# SPEC-004: Plantel cerrado por equipo

- **Issue:** #19
- **Estado:** aprobada (propuesta del usuario, acordada 2026-10-02)
- **ADR relacionados:** ninguno nuevo (cambia reglas de SPEC-001 dentro de ADR-0002)
- **Spike:** no hace falta: no hay incertidumbre técnica (asignación húngara ya en uso); la incertidumbre es de
  producto (intercambios vs. fragmentación) y se evalúa con la revisión visual y luego con S6.

## Problema
Tras S5a la identidad se sigue fragmentando: 14 a 22 IDs por clip de 20 s cuando hay 12–14 jugadores. Cada vez que
la Re-ID duda, se crea una identidad nueva (SPEC-001 RF-8/RF-9, umbral estricto de S4.1). El usuario revisó los
videos y propone usar lo que sabemos del vóley: cada equipo tiene un plantel en cancha acotado.

## Objetivos y alcance
- Dentro: IDs de campo `01`..`06` y `07` para el líbero por equipo; re-identificación por descarte cuando el cupo
  está completo; máximo 6 visibles por equipo; suplentes con ID nuevo solo con evidencia de número.
- Fuera: detectar sustituciones por el árbitro o el marcador; rotaciones; más de un líbero con IDs distintos.

## Requerimientos funcionales
- RF-1: **rol.** El clasificador de equipo informa además si el torso corresponde al prototipo de líbero (color de
  líbero inequívoco, o color ambiguo resuelto como líbero por lado o por número). El tracklet vota el rol; es líbero
  si la mayoría de sus votos lo son. Un número de `libero_numbers` leído vota líbero a través del clasificador
  (SPEC-003 RF-6, solo con color ambiguo). Un tracklet vinculado cuyo rol sostenido contradice al de su identidad
  se desvincula; la partición, la corrección de intercambios y la fusión por número no cruzan roles.
- RF-2: **IDs.** Por equipo: identidades de campo `TEAM_X_PLAYER_01`..`06` (el menor número libre) y líbero
  `TEAM_X_PLAYER_07`. Suplentes (RF-6) desde `08`. Un tracklet solo se asocia a identidades de su mismo rol.
- RF-3: **cupo visible.** Como máximo 6 identidades del equipo observadas en un mismo frame (campo + líbero),
  como hasta ahora (SPEC-001 RF-8).
- RF-4: **descarte.** Cuando el equipo ya tiene sus 6 identidades de campo (o su líbero), un tracklet nuevo con las
  condiciones de creación (en cancha `confirm_frames`, votos de equipo ≥ `team_min_share`) ya no crea identidad:
  se asigna a una identidad **no observada en el frame** del mismo equipo y rol, por asignación húngara de
  apariencia + movimiento, **sin** los umbrales de Re-ID (radio de movimiento, `appearance_only_max_dist`). Si
  hay una sola libre, la toma.
- RF-5: **veto por número.** Un tracklet con número establecido M no puede tomar por descarte una identidad de
  campo con número confirmado N ≠ M. El líbero (`07`) es un rol: no se veta por número (dos líberos que se turnan
  comparten el `07`; el número se muestra solo si el tracklet lo leyó, SPEC-003 RF-8).
- RF-6: **suplente.** Si el descarte no es posible solo por el veto de RF-5 y ninguna identidad del equipo tiene M,
  se crea `TEAM_X_PLAYER_08+` y la identidad de campo que le asignó el húngaro (la que salió) queda **retirada**: no participa
  del descarte, solo vuelve por Re-ID normal (umbrales de SPEC-001).
- RF-6b (revisión S5c M1): un titular retirado que vuelve con su número confirmado recupera su identidad.
- RF-7: mientras el cupo no está completo se mantiene la política anterior (Re-ID con umbrales, si no, identidad
  nueva).

## Requerimientos no funcionales
- RNF-1: determinista; sin dependencias nuevas.
- RNF-2: configurable (`IdentityConfig.closed_roster`, default activado; `field_slots` = 6).

## Casos de uso
- CU-1: un japonés sale de cuadro 10 s y vuelve iluminado distinto: recupera su ID porque es la única libre.
- CU-2: tras un corte de cámara vuelven los 6: cada uno recupera su ID por apariencia entre los 6.

## Reglas de negocio
- RN-1: el `07` es siempre el líbero del equipo; nunca un jugador de campo.
- RN-2: ante duda entre dos ausentes se elige el menor costo; un error es un intercambio, que el número corrige
  (fusión/partición de SPEC-003) cuando se lee.

## Criterios de aceptación
- [ ] AC-1: dado un equipo con 6 identidades de campo y una de ellas ausente 200 frames, cuando vuelve con otra apariencia y lejos de donde se perdió, entonces recupera su `player_id` (no se crea `08`).
- [ ] AC-2: dado dos identidades ausentes y dos tracklets nuevos con apariencias distintas, cuando el cupo está completo, entonces cada uno recupera la identidad de apariencia más parecida.
- [ ] AC-3: dado observaciones marcadas como líbero, cuando se crea su identidad, entonces es `TEAM_X_PLAYER_07`; y un tracklet de campo nunca toma el `07` ni el líbero una de campo.
- [ ] AC-4: dado 6 identidades del equipo observadas en el frame, cuando aparece un séptimo tracklet del equipo, entonces no recibe identidad.
- [ ] AC-5: dado 6 identidades de campo con números confirmados y una ausente (#7), cuando aparece un tracklet que confirma #14 (que nadie tiene), entonces se crea `TEAM_X_PLAYER_08` y la del #7 queda retirada del descarte.
- [ ] AC-6: dado 6 identidades de campo con números confirmados, una ausente (#7) y un tracklet con #9 confirmado que tiene otra identidad visible, cuando se procesa, entonces el tracklet no toma la del #7.
- [ ] AC-7: dado un corte de cámara con 6 + 6 jugadores que vuelven con la misma apariencia, cuando se procesa, entonces no se crea ninguna identidad nueva y cada jugador recupera la suya.
- [ ] AC-8: dado el clasificador de equipo, cuando el torso es el prototipo de líbero (inequívoco o resuelto como líbero), entonces informa rol líbero; si es de color principal, rol campo.
- [ ] AC-10 (revisión S5c): dado una fusión por número antes de completar el cupo, la partición de un tracklet que pasa al líbero, un titular retirado que vuelve con su número y un suplente con otro ausente en juego, cuando se procesan, entonces nunca hay un `07` de campo ni `player_id` duplicados, el líbero toma el `07`, el titular recupera su ID y se retira la identidad que salió.
- [ ] AC-9: dado los clips reales A2, A3, A4, K2, K3 y K5, cuando corre el pipeline, entonces ningún equipo tiene más de 7 IDs salvo suplentes con número, y la revisión visual del usuario aprueba (evidencia en el informe).

## Restricciones técnicas
Cambio en `IdentityManager` (`_associate`, `_create`) y en `TeamClassifier` (rol). El pipeline pasa el rol en la
`Observation`.

## Manejo de errores
- Tracklet sin rol votado: se trata como campo (el líbero es 1 de 7).
- Identidad retirada que vuelve con su número: Re-ID normal; si no pasa los umbrales, se fusiona por número
  (SPEC-003 RF-3) cuando el número se confirma.

## Seguridad
Sin impacto.

## Estrategia de testing
| AC | Test |
|---|---|
| AC-1…AC-7, AC-10 | escenarios de identidad (`tests/scenarios/test_closed_roster.py`) |
| AC-8 | unit de equipo (`tests/unit/test_team_roles.py`) |
| AC-9 | revisión visual manual (usuario) |

Cambios de semántica respecto de SPEC-001/S4.1, aprobados por el usuario: con el cupo completo, una persona nueva
ya no recibe un ID nuevo sino uno libre por descarte. Los tests que fijaban la semántica anterior
(`test_ac21_full_roster_with_occluded_identity_admits_real_player`, `test_finding6_reset_keeps_lost_and_frees_roster`)
siguen corriendo con `closed_roster=False`; la nueva semántica la cubren AC-1…AC-7. AC-10 de SPEC-001 (sustitución)
no cambia: con un solo jugador el cupo no está completo (RF-7).

## Dependencias e impacto
Modifica `volley_cv.identity` (tipos, settings, manager), `volley_cv.team` y `volley_cv.pipeline`. Las demás
reglas de SPEC-001 y SPEC-003 siguen vigentes.
