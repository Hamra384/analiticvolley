---
name: adr
description: Registra una decisión arquitectónica en docs/adr/ (contexto, decisión, alternativas, consecuencias, riesgos). Solo para decisiones no triviales.
---

1. ¿Es arquitectónica? (cambia límites entre componentes, una dependencia central, un formato de datos, un
   contrato). Si no, no crees ADR.
2. Copiá `docs/adr/0000-template.md` a `docs/adr/NNNN-<slug>.md` con el siguiente número libre.
3. El contexto se apoya en hechos medidos (spike, auditoría); marcá las hipótesis.
4. Listá al menos dos alternativas reales con pros y contras.
5. Definí el criterio de revisión: qué evidencia haría reabrir la decisión.
6. Si reemplaza a otro ADR, actualizá el estado del anterior a "reemplazado por ADR-NNNN".
