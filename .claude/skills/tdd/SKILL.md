---
name: tdd
description: Ciclo Red-Green-Refactor para implementar los criterios de aceptación de un Issue. Usar para todo código de producción en src/.
---

Para cada AC del Issue, en orden:

1. **RED**: escribí el test (o delegá al subagente `qa-engineer`). Corré `uv run pytest <archivo> -q`.
   Confirmá que falla **por la razón esperada**: la aserción sobre el comportamiento que falta. Un
   ImportError o un typo no cuentan como RED válido: corregí el test y volvé a correr.
2. **GREEN**: escribí el mínimo código en `src/` para que pase. Corré el test y luego toda la suite.
3. **REFACTOR**: limpiá sin cambiar comportamiento. La suite sigue en verde. `uv run ruff check . && uv run mypy`.
4. Commit chico: `git commit -m "<qué> (Refs #N)"`.

Reglas:
- Nunca modifiques un test para que pase el código; si el test estaba mal, explicá por qué en el commit.
- Nada de skip/xfail sin `#Issue`. No borres asserts.
- Registrá en el informe la salida RED y GREEN de cada AC (evidencia).
