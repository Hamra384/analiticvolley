---
spike: "003"
issue: "#11"
fecha: 2026-10-02
estado: cerrado
---

# SPIKE-003: ¿Qué descriptor de apariencia distingue mejor a jugadores reales?

## Pregunta
El IdentityManager re-identifica jugadores por apariencia. En S4 se usó ResNet18 ImageNet sin validarlo con datos
reales; en el debug de A2 se vieron mezclas entre compañeros de equipo. ¿Sirve? ¿Hay algo mejor?

## Criterio de decisión (fijado antes de medir)
Gana el descriptor con mayor AUC en el subgrupo **difícil** (compañeros con torso de color parecido), promediado en
A2 y K2. Desempate: costo (GPU, velocidad) y licencia.

## Método
`spikes/appearance/eval_embeddings.py` sobre los tracks ByteTrack cacheados de A2 y K2 (personas de altura ≥ 7 %).
- **Positivos**: mismo track a 10 frames de distancia (400 pares por clip).
- **Negativos**: tracks distintos en el mismo frame (1600 pares); **difíciles**: distancia Lab de torso < 20
  (223 en A2, 167 en K2).
- AUC = P(distancia del negativo > distancia del positivo).

## Resultados

| Descriptor | A2 AUC general | A2 AUC difícil | K2 AUC general | K2 AUC difícil | Mismo jugador p90 (A2/K2) | Difícil mediana (A2/K2) |
|---|---|---|---|---|---|---|
| **Histograma HSV 12×6×6 (Hellinger)** | **0,988** | **0,965** | **0,988** | **0,929** | 0,141 / 0,141 | 0,373 / 0,252 |
| OSNet x0.25 MSMT17 (boxmot) | 0,983 | 0,953 | 0,982 | 0,903 | 0,253 / 0,215 | 0,433 / 0,292 |
| ResNet18 ImageNet (torchvision) | 0,967 | 0,921 | 0,971 | 0,873 | 0,183 / 0,193 | 0,261 / 0,216 |

## Decisión
**Histograma de color** como descriptor del pipeline: mejor AUC difícil en ambos clips, sin GPU, sin pesos con
licencia no comercial (MSMT17). ResNet18 era el peor de los tres.

Umbrales recalibrados con estas distribuciones: `split_distance` 0,35 → **0,20** (mismo jugador p90 = 0,14 <
0,20 < compañeros distintos mediana 0,25–0,37); `appearance_only_max_dist` = 0,08 (< p10 de difíciles, 0,095–0,12).

## Costo asumido del umbral estricto (revisión S4.1, M2)
`appearance_only_max_dist = 0,08` queda **por debajo** del p90 del mismo jugador incluso en el caso fácil (0,14
a 10 frames): rechaza > 10 % de re-identificaciones correctas allí y probablemente la mayoría tras un corte o un
hueco largo. En la práctica, **la Re-ID solo por apariencia queda casi desactivada**: se prefiere una identidad
nueva (fragmentación, visible en K2: 11 + 9 identidades) antes que heredar la de otra persona (los errores que vio
el usuario). La señal que debe reemplazarla es el número de camiseta (S5a). Medir el costo real requiere S6.

## Limitaciones (honestas)
- Los positivos están a 10 frames (misma pose y luz): favorece al color. Para re-identificar tras huecos largos o
  cortes de cámara la ventaja puede achicarse. Revalidar en S6 con ground truth.
- Los "difíciles" se definen por color de torso parecido, no por identidad real de equipo.
- OSNet se probó con pesos genéricos de personas; uno ajustado a vóley podría ganar (fuera de alcance).
