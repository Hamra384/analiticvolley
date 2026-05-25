import cv2
import easyocr
import numpy as np
from collections import deque
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=True)

source = "D:/AIVolley/analiticvolley/videos/partido.mp4"

cap = cv2.VideoCapture(source)

minutos = 14
segundos = 10
fps = cap.get(cv2.CAP_PROP_FPS)
frame_inicio = int((minutos * 60 + segundos) * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

altura_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ancho_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
LIMITE_Y_RED_FALLBACK = int(altura_frame * 0.38)

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
KERNEL_SHARP = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])

frame_count = 0
numero_confirmado = {}
votos = {}
y_ataque_izq = None
y_ataque_der = None
y_red_izq = None
y_red_der = None
y_central_izq = None
y_central_der = None
y_ataque_izq_hist = deque(maxlen=45)
y_ataque_der_hist = deque(maxlen=45)
# ratio: distancia ataque->red / distancia ataque->fondo del frame  (+/-)
ratio_ataque_red = 0.53
# ratio: distancia ataque->linea central piso  ([/])
ratio_central_piso = 0.18

MARGEN_AMBIGUO = 60  # px: zona gris alrededor de la línea central donde se usa color

lado_historial = {}     # tid -> deque(maxlen=15) de bool
color_jugador = {}      # tid -> np.array BGR (media EMA del torso)
colores_cercano = deque(maxlen=300)
colores_lejano = deque(maxlen=300)
centroide_cercano = None
centroide_lejano = None

ultima_pos = {}         # tid -> (cx, cy) última posición conocida
ids_prev_frame = set()
jugadores_perdidos = {} # tid -> {'numero', 'frame', 'cx', 'cy'}
FRAMES_HERENCIA = 90    # ~3 s a 30 fps
DIST_HERENCIA = 180     # px: distancia máxima para heredar número
frame_aparicion = {}    # tid -> primer frame en que se vio


def _y_ataque_en_franja(frame, x_ini, x_fin):
    """Detecta la Y de la línea de ataque en una franja vertical del frame."""
    h = frame.shape[0]
    y_min = int(h * 0.38)
    y_max = int(h * 0.90)
    franja = frame[y_min:y_max, x_ini:x_fin]
    ancho_franja = x_fin - x_ini

    gris = cv2.cvtColor(franja, cv2.COLOR_BGR2GRAY)
    _, blanco = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)

    lineas = cv2.HoughLinesP(blanco, 1, np.pi / 180,
                              threshold=60,
                              minLineLength=ancho_franja * 0.35,
                              maxLineGap=40)
    if lineas is None:
        return None

    candidatas = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        angulo = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        y_media = (y1 + y2) // 2 + y_min
        if angulo < 5 and y_media < h * 0.78:
            candidatas.append(y_media)

    if not candidatas:
        return None

    return min(candidatas)  # la más cerca de la red (menor Y en la zona de piso)


def detectar_linea_ataque(frame):
    """Detecta la línea de ataque como línea inclinada detectando cada mitad por separado."""
    w = frame.shape[1]
    mitad = w // 2

    y_izq = _y_ataque_en_franja(frame, 0, mitad)
    y_der = _y_ataque_en_franja(frame, mitad, w)

    if y_izq is None and y_der is None:
        return None
    if y_izq is None:
        y_izq = y_der
    if y_der is None:
        y_der = y_izq

    return y_izq, y_der


def detectar_red(frame):
    """Estima la línea de red y actualiza la línea central del piso."""
    global y_ataque_izq, y_ataque_der, y_red_izq, y_red_der
    global y_central_izq, y_central_der

    h = frame.shape[0]
    resultado = detectar_linea_ataque(frame)

    if resultado is not None:
        yi, yd = resultado
        y_ataque_izq_hist.append(yi)
        y_ataque_der_hist.append(yd)
        y_ataque_izq = int(np.median(y_ataque_izq_hist))
        y_ataque_der = int(np.median(y_ataque_der_hist))

    if y_ataque_izq is not None:
        y_red_izq = int(y_ataque_izq - (h - y_ataque_izq) * ratio_ataque_red)
        y_red_der = int(y_ataque_der - (h - y_ataque_der) * ratio_ataque_red)

    if y_ataque_izq is not None:
        y_central_izq = int(y_ataque_izq - (h - y_ataque_izq) * ratio_central_piso)
        y_central_der = int(y_ataque_der - (h - y_ataque_der) * ratio_central_piso)

    return (y_red_izq, y_red_der), (y_ataque_izq, y_ataque_der)


def y_red_en_x(x):
    """Devuelve la Y de la línea de red en una coordenada X dada."""
    if y_red_izq is None:
        return LIMITE_Y_RED_FALLBACK
    t = x / ancho_frame
    return int(y_red_izq + (y_red_der - y_red_izq) * t)


def y_central_en_x(x):
    """Devuelve la Y de la línea central del piso en una coordenada X dada."""
    if y_central_izq is None:
        return LIMITE_Y_RED_FALLBACK
    t = x / ancho_frame
    return int(y_central_izq + (y_central_der - y_central_izq) * t)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    correr_ocr = (frame_count % 15 == 0)

    results = model.track(
        frame,
        classes=[0],
        persist=True,
        tracker="bytetrack_custom.yaml",
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    (yr_izq, yr_der), (ya_izq, ya_der) = detectar_red(frame)

    # Actualizar centroides de color y limpiar jugadores_perdidos viejos cada 60 frames
    if frame_count % 60 == 0:
        if len(colores_cercano) > 15 and len(colores_lejano) > 15:
            centroide_cercano = np.mean(colores_cercano, axis=0)
            centroide_lejano = np.mean(colores_lejano, axis=0)
        jugadores_perdidos_limpio = {
            k: v for k, v in jugadores_perdidos.items()
            if frame_count - v['frame'] <= FRAMES_HERENCIA
        }
        jugadores_perdidos.clear()
        jugadores_perdidos.update(jugadores_perdidos_limpio)

    annotated = frame.copy()
    ids_este_frame = set()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            alto = y2 - y1
            ancho = x2 - x1
            tid = int(track_id)

            centro_x = (x1 + x2) // 2
            cy_box = (y1 + y2) // 2
            ultima_pos[tid] = (centro_x, cy_box)
            ids_este_frame.add(tid)

            # Registrar primer frame en que se ve este TID
            if tid not in frame_aparicion:
                frame_aparicion[tid] = frame_count

            # Herencia: solo para TIDs recién aparecidos (máx 5 frames de vida)
            # y tomar el jugador perdido MÁS CERCANO dentro del umbral, no el primero
            es_tid_nuevo = (frame_count - frame_aparicion[tid]) <= 5
            if tid not in numero_confirmado and es_tid_nuevo:
                mejor_tid_p = None
                mejor_dist = DIST_HERENCIA
                for tid_p, datos_p in list(jugadores_perdidos.items()):
                    if frame_count - datos_p['frame'] > FRAMES_HERENCIA:
                        jugadores_perdidos.pop(tid_p, None)
                        continue
                    dist = np.hypot(centro_x - datos_p['cx'], cy_box - datos_p['cy'])
                    if dist < mejor_dist:
                        mejor_dist = dist
                        mejor_tid_p = tid_p
                if mejor_tid_p is not None:
                    datos_p = jugadores_perdidos.pop(mejor_tid_p)
                    numero_confirmado.pop(mejor_tid_p, None)
                    numero_confirmado[tid] = datos_p['numero']
                    votos.pop(mejor_tid_p, None)
                    print(f"ID {tid} heredó #{datos_p['numero']} de ID {mejor_tid_p} (dist={mejor_dist:.0f})")

            y_central = y_central_en_x(centro_x)
            diferencia = y2 - y_central  # positivo = pies debajo de la línea = cercano

            # Muestrear color del torso (EMA) si el box es suficientemente grande
            if ancho > 40 and alto > 80:
                tx1 = x1 + int(ancho * 0.2)
                tx2 = x2 - int(ancho * 0.2)
                ty1 = y1 + int(alto * 0.15)
                ty2 = y1 + int(alto * 0.45)
                torso = frame[ty1:ty2, tx1:tx2]
                if torso.size > 0:
                    color_actual = np.mean(torso.reshape(-1, 3), axis=0).astype(float)
                    if tid not in color_jugador:
                        color_jugador[tid] = color_actual
                    else:
                        color_jugador[tid] = 0.85 * color_jugador[tid] + 0.15 * color_actual
                    # Entrenar centroides con jugadores inequívocamente de un lado
                    if diferencia > MARGEN_AMBIGUO:
                        colores_cercano.append(color_jugador[tid].copy())
                    elif diferencia < -MARGEN_AMBIGUO:
                        colores_lejano.append(color_jugador[tid].copy())

            # Clasificar: posición clara → posición; zona ambigua → color
            if abs(diferencia) > MARGEN_AMBIGUO:
                es_cercano_raw = diferencia > 0
            elif (centroide_cercano is not None and centroide_lejano is not None
                  and tid in color_jugador):
                d_c = np.linalg.norm(color_jugador[tid] - centroide_cercano)
                d_l = np.linalg.norm(color_jugador[tid] - centroide_lejano)
                es_cercano_raw = d_c < d_l
            else:
                es_cercano_raw = diferencia > 0

            # Histéresis: mayoría de los últimos 15 frames decide
            if tid not in lado_historial:
                lado_historial[tid] = deque(maxlen=15)
            lado_historial[tid].append(es_cercano_raw)
            es_cercano = sum(lado_historial[tid]) > len(lado_historial[tid]) / 2

            if not es_cercano:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue

            if correr_ocr and tid not in numero_confirmado and ancho > 40 and alto > 80:
                margen_x = int(ancho * 0.05)
                x1r = max(0, x1 - margen_x)
                x2r = min(ancho_frame, x2 + margen_x)
                recorte = frame[y1 + int(alto * 0.05):y1 + int(alto * 0.65), x1r:x2r]

                if recorte.size == 0:
                    pass
                else:
                    # Escalar: apuntar a ~200px de altura
                    escala = max(3, 200 // recorte.shape[0])
                    recorte_g = cv2.resize(
                        recorte,
                        (recorte.shape[1] * escala, recorte.shape[0] * escala),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    h_g, w_g = recorte_g.shape[:2]

                    gris = cv2.cvtColor(recorte_g, cv2.COLOR_BGR2GRAY)
                    gris_clahe = clahe.apply(gris)

                    # Intentar aislar la región del número con connected components
                    # Probar ambas polaridades (número claro sobre oscuro y viceversa)
                    mejor_crop = None
                    for thresh_flag in [cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
                        _, binaria = cv2.threshold(gris_clahe, 0, 255, thresh_flag)
                        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, ker)

                        n, _, stats, _ = cv2.connectedComponentsWithStats(binaria)
                        area_total = h_g * w_g
                        digitos = []
                        for i in range(1, n):
                            cx, cy, cw, ch, ca = stats[i]
                            if cw == 0:
                                continue
                            aspecto = ch / cw
                            rel = ca / area_total
                            centro_x = cx + cw / 2
                            # Componente con forma de dígito, tamaño razonable, centrado
                            if (1.0 < aspecto < 4.5
                                    and 0.02 < rel < 0.45
                                    and w_g * 0.05 < centro_x < w_g * 0.95):
                                digitos.append((cx, cy, cw, ch, ca))

                        if not digitos:
                            continue

                        # Tomar los 1-2 más grandes
                        digitos.sort(key=lambda d: d[4], reverse=True)
                        sel = sorted(digitos[:2], key=lambda d: d[0])

                        xm = min(d[0] for d in sel)
                        ym = min(d[1] for d in sel)
                        xM = max(d[0] + d[2] for d in sel)
                        yM = max(d[1] + d[3] for d in sel)

                        pad = int((yM - ym) * 0.25)
                        xm = max(0, xm - pad)
                        ym = max(0, ym - pad)
                        xM = min(w_g, xM + pad)
                        yM = min(h_g, yM + pad)

                        crop_cand = recorte_g[ym:yM, xm:xM]
                        if crop_cand.size > 0:
                            mejor_crop = crop_cand
                            break

                    gris_sharp = np.clip(
                        cv2.filter2D(gris, -1, KERNEL_SHARP), 0, 255
                    ).astype(np.uint8)

                    # Imágenes a intentar: número aislado primero, luego torso completo
                    imgs_ocr = []
                    if mejor_crop is not None:
                        mc_g = cv2.cvtColor(mejor_crop, cv2.COLOR_BGR2GRAY)
                        imgs_ocr += [mejor_crop, mc_g, cv2.bitwise_not(mc_g)]
                    imgs_ocr += [recorte_g, gris_sharp, cv2.bitwise_not(gris_clahe)]

                    candidatos = []
                    for img in imgs_ocr:
                        r = reader.readtext(
                            img,
                            allowlist='0123456789',
                            min_size=6,
                            text_threshold=0.3,
                            low_text=0.15,
                            width_ths=0.9,
                        )
                        candidatos.extend(r)

                    if candidatos:
                        print(f"ID {tid}: {[(c[1], round(c[2],2)) for c in candidatos]}")
                        mejor = max(candidatos, key=lambda x: x[2])
                        texto = mejor[1].strip()
                        confianza = mejor[2]
                        numeros_activos = {
                            n for t, n in numero_confirmado.items() if t != tid
                        }
                        if (texto.isdigit() and 1 <= len(texto) <= 2
                                and confianza > 0.38
                                and texto not in numeros_activos):
                            # Confirmación inmediata con confianza muy alta
                            if confianza >= 0.88:
                                numero_confirmado[tid] = texto
                            else:
                                if tid not in votos:
                                    votos[tid] = {}
                                votos[tid][texto] = votos[tid].get(texto, 0) + 1
                                votos_necesarios = 3 if len(texto) == 2 else 2
                                if votos[tid][texto] >= votos_necesarios:
                                    numero_confirmado[tid] = texto

            numero = numero_confirmado.get(tid, "")
            color = (0, 255, 0) if numero else (0, 150, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{numero}" if numero else f"ID:{tid}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Guardar IDs que desaparecieron — solo si tienen votos OCR reales (no heredados)
    for tid_perdido in ids_prev_frame - ids_este_frame:
        if (tid_perdido in numero_confirmado
                and tid_perdido not in jugadores_perdidos
                and votos.get(tid_perdido)):
            pos = ultima_pos.get(tid_perdido, (ancho_frame // 2, altura_frame // 2))
            jugadores_perdidos[tid_perdido] = {
                'numero': numero_confirmado[tid_perdido],
                'frame': frame_count,
                'cx': pos[0],
                'cy': pos[1],
            }
    ids_prev_frame = ids_este_frame

    # Línea de ataque (cyan), red estimada (amarillo), central del piso (verde)
    if ya_izq is not None:
        cv2.line(annotated, (0, ya_izq), (ancho_frame, ya_der), (255, 200, 0), 1)
    if yr_izq is not None:
        cv2.line(annotated, (0, yr_izq), (ancho_frame, yr_der), (0, 255, 255), 2)
    if y_central_izq is not None:
        cv2.line(annotated, (0, y_central_izq), (ancho_frame, y_central_der), (0, 255, 0), 2)
    cv2.putText(annotated, f"red ratio: {ratio_ataque_red:.2f} (+/-)  central: {ratio_central_piso:.2f} ([/])",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Tracking", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == 43:
        ratio_ataque_red = round(min(ratio_ataque_red + 0.02, 2.0), 2)
    elif key == ord('-') or key == 45:
        ratio_ataque_red = round(max(ratio_ataque_red - 0.02, 0.05), 2)
    elif key == ord('['):
        ratio_central_piso = round(max(ratio_central_piso - 0.02, 0.05), 2)
    elif key == ord(']'):
        ratio_central_piso = round(min(ratio_central_piso + 0.02, 2.0), 2)

cap.release()
cv2.destroyAllWindows()