# Informe de sprint S5c — Plantel cerrado por equipo

- **Fecha:** 2026-10-02 · **Issue:** #19 · **Rama:** `19-s5c-closed-roster` · **Spec:** SPEC-004
- **Origen:** propuesta del usuario tras revisar los videos de S5a ("se sigue confundiendo muchas veces"): IDs por
  equipo del 1 al 7, el 7 siempre el líbero, máximo 6 por equipo en cancha, y que un jugador que se pierde vuelva
  con el mismo ID por descarte en vez de crear IDs nuevos.

## Qué se hizo

1. **Rol de líbero** en el clasificador de equipo (`classify_roles`): color de líbero inequívoco, o color ambiguo
   resuelto como líbero por lado o por número. El pipeline lo pasa en la `Observation`.
2. **IDs cerrados**: campo `01`..`06` (el menor libre), líbero `07`, suplentes `08+` solo si el número de camiseta
   demuestra que es otra persona (acordado con el usuario).
3. **Re-identificación por descarte**: con el cupo completo, un jugador que aparece toma una identidad ausente de
   su equipo y rol (asignación húngara por apariencia y movimiento), sin los umbrales estrictos de S4.1.
4. **Veto por número**: un número establecido distinto impide tomar esa identidad por descarte.
5. **Tope de 6 visibles por equipo** en todos los caminos (vinculados, Re-ID, descarte).

## Resultados en video real (sin ground truth)

| Clip | IDs S5a | IDs S5c | Máx. visibles por equipo | `player_id` duplicados |
|---|---|---|---|---|
| A2 | 16 | 13 (7 + 6) | 6 / 4 | 0 |
| A3 | 14 | 14 (7 + 7) | 6 / 6 | 0 |
| A4 | 17 | 14 (7 + 7) | 5 / 6 | 0 |
| K2 | 22 | 14 (7 + 7) | 6 / 6 | 0 |
| K3 | 15 | 14 (7 + 7) | 6 / 6 | 0 |
| K5 | 19 | 14 (7 + 7) | 6 / 6 | 0 |

- **Revisión visual del usuario: aprobada** ("está perfecto").
- El líbero argentino #19 queda como `B07` en A3 (471 frames) y A4 (425 frames).
- **Sigue mal (hecho, revisado a ojo):** en A2, `B07` toma al principio (frames 37-92) a japoneses de blanco cerca de la
  red (#35, #24) antes de ser el líbero #19 real desde el frame 265. Es el problema de lado ambiguo cerca de la red
  (#13): el plantel cerrado no lo resuelve. En A2 también hay un `A07` de 2 frames (alguien de Japón marcado como
  líbero un instante).
- Costo asumido: los errores que quedan son **intercambios** de ID entre jugadores ausentes en vez de IDs nuevos.
  Su frecuencia es **desconocida** sin anotación (S6).

## Revisión independiente (subagente reviewer)

2 altos y 2 medios, todos reproducidos y corregidos con test de regresión (RED observado):
- **H1** tras una fusión por número antes de completar el cupo, un jugador de campo podía recibir el `07` y pisar al
  líbero (`player_id` duplicado) → numeración del menor libre, `07` solo para el líbero, y control de duplicados.
- **H2** si el tracker pasaba el track de un jugador de campo al líbero, el fragmento partido perdía el rol y el
  líbero se quedaba con un ID de campo → el fragmento hereda su rol; los roles no se cruzan al vincular, corregir
  intercambios ni fusionar; un tracklet con rol contradicho se desvincula.
- **M1** un titular reemplazado por un suplente no recuperaba su ID al volver → lo recupera por su número.
- **M2** el suplente podía retirar al jugador equivocado → retira la identidad que le asignó el húngaro.
- Bajos: tope visible calculado después de las contradicciones, la Re-ID normal reactiva retiradas, spec aclarada
  (RF-1, RF-6b), test de AC-5 más estricto.

Además, en mi verificación con video real encontré que el tope de 6 no cubría la re-identificación (7 visibles en
K2/K5); corregido con test antes de la revisión.

## Pruebas

- 343 tests locales (0 salteados); ruff, format y mypy limpios. CI: ver PR.
- Dos tests de la semántica anterior (persona nueva con el cupo lleno → ID nuevo) corren con `closed_roster=False`;
  es el cambio de semántica aprobado por el usuario.

## Errores míos en este sprint

- El primer tope de 6 solo cubría tracks ya vinculados; lo detecté en video real.
- Una corrida de A3 se cortó en el frame 538 sin error visible (hipótesis: memoria o el video abierto en el
  reproductor); repetida completa con el mismo código.
- Un test asumía un escenario que no reproducía el caso real (el `06` nunca llegaba a existir).

## Riesgos y pendientes

- Lado ambiguo cerca de la red (#13): contamina al `07` de Argentina en A2.
- Si al principio del clip se crean identidades con fragmentos de la misma persona, el cupo se llena antes y los
  jugadores que aparecen después toman IDs por descarte (posibles intercambios).
- Siguen #14 (mezclas), #15 (oficiales/gráficos), #16 (velocidad: ~5 FPS en 1080p, ~8,5 en 720p).

## Próximo

S5b: pelota (detector específico y estados DETECTED/PREDICTED/LOST). Para S6 hace falta la anotación en CVAT.
