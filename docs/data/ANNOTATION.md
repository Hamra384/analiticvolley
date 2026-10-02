# Guía de anotación (corrección en CVAT)

**Para qué sirve:** para medir si el sistema mantiene bien la identidad de cada jugador necesitamos la "respuesta
correcta" de cada clip. El sistema ya hizo un primer borrador automático; tu trabajo es **corregirlo**, no dibujar
desde cero.

- 10 clips de 20 segundos (600 imágenes cada uno).
- Tiempo estimado: 25–40 minutos por clip. No hace falta hacerlos todos de una vez: CVAT guarda el avance.
- Orden sugerido: A2, K2, A1, K1 primero (los más útiles), después el resto.

## Abrir CVAT

1. Abrí **http://localhost:8080** en el navegador (Docker Desktop tiene que estar abierto).
2. Usuario y contraseña: te los paso en el chat.
3. Entrá a **Tasks**. Cada clip es una tarea (`A1`, `A2`, …). Abrí la tarea y hacé clic en **Job #…**.

## Qué corregir, en orden de importancia

### 1. Que cada jugador tenga UN solo número de track en todo el clip (lo más importante)

Cada caja tiene un número (por ejemplo `player 12`). Ese número tiene que representar **a la misma persona real
de principio a fin**, aunque se tape, se cruce con otro o desaparezca un momento.

El borrador automático suele fallar así:
- **Un jugador con varios números** (el track se cortó). Arreglo: seleccioná los dos tracks y usá **Merge**
  (tecla `M`), así quedan como uno solo.
- **Un número que salta de un jugador a otro** (se mezclaron). Arreglo: andá al frame donde salta, usá
  **Split** (tecla `Alt+M`) para cortar el track, y después hacé Merge de cada pedazo con la persona correcta.

### 2. Borrar lo que no es un jugador en cancha

Borrá (tecla `Supr`) los tracks de árbitros, jueces de línea, banco, entrenadores y público. El **líbero sí cuenta**
como jugador aunque tenga otra camiseta.

### 3. Agregar jugadores que faltan

Si un jugador en cancha no tiene caja, dibujala (tecla `N`) como **track** con la etiqueta `player`. CVAT
interpola entre los frames que marques: alcanza con ajustar la caja cada ~10 frames.

### 4. Completar los atributos de cada jugador (una vez por track)

- **team**: `A` = el equipo que empieza **abajo** (lado cercano a la cámara) en ese clip, `B` = el de arriba.
- **jersey**: el número de camiseta si lo podés leer en algún momento. Si no se lee nunca, dejalo vacío
  (**no adivines**: vacío es la respuesta correcta).
- **libero**: marcalo si es el líbero.

### 5. La pelota

Hay un track con la etiqueta `ball`. Corregilo para que la caja esté sobre la pelota en cada frame donde **se ve**.
Donde no se ve (tapada o fuera de cuadro), marcá el track como **outside** (tecla `O`) para esos frames.

## Atajos útiles

(Si algún atajo no responde, la misma acción está en el menú contextual de la caja, clic derecho.)

| Tecla | Acción |
|---|---|
| `F` / `D` | frame siguiente / anterior |
| `V` / `C` | avanzar / retroceder 10 frames |
| `N` | dibujar caja nueva |
| `M` | merge (unir tracks) |
| `Alt+M` | split (cortar un track) |
| `O` | marcar outside (no visible) |
| `Ctrl+S` | **guardar** (hacelo seguido) |

## Cuando termines un clip

Guardá (`Ctrl+S`) y avisame en el chat ("terminé A2"). Yo exporto las anotaciones desde CVAT; no tenés que
descargar nada.

## Dudas frecuentes

- *¿Y si no sé si dos imágenes son el mismo jugador?* Usá el número de camiseta, el pelo, las rodilleras, las
  zapatillas. Si de verdad no se puede saber, dejalo como está y anotalo en el chat.
- *¿Tengo que ser exacto con los bordes de las cajas?* No: que la caja cubra al jugador es suficiente. Lo que
  importa es **que la identidad sea correcta**.
