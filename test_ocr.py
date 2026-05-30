import cv2
import numpy as np
import threading
import os
from collections import deque
from ultralytics import YOLO
import easyocr

model  = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=True)

# ── OCR asíncrono ─────────────────────────────────────────────────────────────
_pending  = {}
_results  = {}
_ocr_cond = threading.Condition()

def _ocr_worker():
    while True:
        with _ocr_cond:
            while not _pending:
                _ocr_cond.wait()
            tid, imgs = next(iter(_pending.items()))
            del _pending[tid]
        candidatos = []
        for img in imgs:
            try:
                r = reader.readtext(img, allowlist='0123456789',
                                    min_size=6, text_threshold=0.3,
                                    low_text=0.15, width_ths=0.9)
                candidatos.extend(r)
            except Exception:
                pass
        if candidatos:
            mejor = max(candidatos, key=lambda x: x[2])
            with _ocr_cond:
                _results[tid] = (mejor[1].strip(), mejor[2])

threading.Thread(target=_ocr_worker, daemon=True).start()

# ── Video ─────────────────────────────────────────────────────────────────────
source = "D:/AIVolley/analiticvolley/videos/partido2.mp4"
cap    = cv2.VideoCapture(source)

minutos  = 3
segundos = 8
fps      = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int((minutos * 60 + segundos) * fps))

altura_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ancho_frame  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
LIMITE_Y_RED_FALLBACK = int(altura_frame * 0.38)

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))

# ── Estado ────────────────────────────────────────────────────────────────────
frame_count        = 0
numero_confirmado  = {}
votos              = {}

y_ataque_izq = y_ataque_der = None
y_red_izq    = y_red_der    = None
y_central_izq = y_central_der = None
y_ataque_izq_hist  = deque(maxlen=45)
y_ataque_der_hist  = deque(maxlen=45)
ratio_ataque_red   = 0.53
ratio_central_piso = 0.18
MARGEN_AMBIGUO     = 60

lado_historial     = {}
color_jugador      = {}
colores_cercano    = deque(maxlen=300)
colores_lejano     = deque(maxlen=300)
centroide_cercano  = None
centroide_lejano   = None

ultima_pos         = {}
ids_prev_frame     = set()
jugadores_perdidos = {}
FRAMES_HERENCIA    = 90
DIST_HERENCIA      = 180
frame_aparicion    = {}
ultimo_ocr_frame   = {}
OCR_COOLDOWN       = 3
es_cercano_dict    = {}

# Pelota: YOLO class 32 cada N frames
bola_pos  = None
bola_tick = 0
BOLA_CADA = 5


# ── Líneas de cancha ──────────────────────────────────────────────────────────
def _y_ataque_en_franja(frame, x_ini, x_fin):
    h = frame.shape[0]
    y_min = int(h * 0.38)
    y_max = int(h * 0.90)
    franja = frame[y_min:y_max, x_ini:x_fin]
    gris = cv2.cvtColor(franja, cv2.COLOR_BGR2GRAY)
    _, blanco = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)
    lineas = cv2.HoughLinesP(blanco, 1, np.pi / 180,
                              threshold=60,
                              minLineLength=(x_fin - x_ini) * 0.35,
                              maxLineGap=40)
    if lineas is None:
        return None
    candidatas = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        angulo  = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        y_media = (y1 + y2) // 2 + y_min
        if angulo < 5 and y_media < h * 0.78:
            candidatas.append(y_media)
    return min(candidatas) if candidatas else None


def detectar_linea_ataque(frame):
    w     = frame.shape[1]
    mitad = w // 2
    y_izq = _y_ataque_en_franja(frame, 0, mitad)
    y_der = _y_ataque_en_franja(frame, mitad, w)
    if y_izq is None and y_der is None:
        return None
    if y_izq is None: y_izq = y_der
    if y_der is None: y_der = y_izq
    return y_izq, y_der


def detectar_red(frame):
    global y_ataque_izq, y_ataque_der, y_red_izq, y_red_der
    global y_central_izq, y_central_der
    h         = frame.shape[0]
    resultado = detectar_linea_ataque(frame)
    if resultado is not None:
        yi, yd = resultado
        y_ataque_izq_hist.append(yi)
        y_ataque_der_hist.append(yd)
        y_ataque_izq = int(np.median(y_ataque_izq_hist))
        y_ataque_der = int(np.median(y_ataque_der_hist))
    if y_ataque_izq is not None:
        y_red_izq     = int(y_ataque_izq - (h - y_ataque_izq) * ratio_ataque_red)
        y_red_der     = int(y_ataque_der - (h - y_ataque_der) * ratio_ataque_red)
        y_central_izq = int(y_ataque_izq - (h - y_ataque_izq) * ratio_central_piso)
        y_central_der = int(y_ataque_der - (h - y_ataque_der) * ratio_central_piso)
    return (y_red_izq, y_red_der), (y_ataque_izq, y_ataque_der)


def y_red_en_x(x):
    if y_red_izq is None:
        return LIMITE_Y_RED_FALLBACK
    t = x / ancho_frame
    return int(y_red_izq + (y_red_der - y_red_izq) * t)


def y_central_en_x(x):
    if y_central_izq is None:
        return LIMITE_Y_RED_FALLBACK
    t = x / ancho_frame
    return int(y_central_izq + (y_central_der - y_central_izq) * t)


# ── Loop principal ────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Consumir resultados del worker OCR
    with _ocr_cond:
        res_frame = dict(_results)
        _results.clear()
    for tid_r, (texto, confianza) in res_frame.items():
        if tid_r in numero_confirmado:
            continue
        numeros_activos = {n for t, n in numero_confirmado.items() if t != tid_r}
        if texto.isdigit() and 1 <= len(texto) <= 2 and confianza > 0.38 and texto not in numeros_activos:
            if confianza >= 0.88:
                numero_confirmado[tid_r] = texto
            else:
                votos.setdefault(tid_r, {})
                votos[tid_r][texto] = votos[tid_r].get(texto, 0) + 1
                if votos[tid_r][texto] >= (3 if len(texto) == 2 else 2):
                    numero_confirmado[tid_r] = texto

    # Tracking de personas
    results = model.track(frame, classes=[0], persist=True,
                          tracker='bytetrack_custom.yaml',
                          imgsz=640, conf=0.25, verbose=False)

    (yr_izq, yr_der), (ya_izq, ya_der) = detectar_red(frame)

    # Detección de pelota (YOLO class 32) cada BOLA_CADA frames
    bola_tick += 1
    if bola_tick >= BOLA_CADA:
        bola_tick = 0
        res_bola = model.predict(frame, classes=[32], conf=0.15, imgsz=640, verbose=False)
        if res_bola[0].boxes is not None and len(res_bola[0].boxes) > 0:
            idx     = int(res_bola[0].boxes.conf.argmax())
            bx      = res_bola[0].boxes.xyxy[idx].cpu().numpy().astype(int)
            bola_pos = ((bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2)
        else:
            bola_pos = None

    # Actualizar centroides de color cada 60 frames
    if frame_count % 60 == 0:
        if len(colores_cercano) > 15 and len(colores_lejano) > 15:
            centroide_cercano = np.mean(colores_cercano, axis=0)
            centroide_lejano  = np.mean(colores_lejano,  axis=0)
        jugadores_perdidos = {k: v for k, v in jugadores_perdidos.items()
                              if frame_count - v['frame'] <= FRAMES_HERENCIA}

    annotated      = frame.copy()
    ids_este_frame = set()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            alto  = y2 - y1
            ancho = x2 - x1
            tid   = int(track_id)

            centro_x = (x1 + x2) // 2
            cy_box   = (y1 + y2) // 2
            ultima_pos[tid] = (centro_x, cy_box)
            ids_este_frame.add(tid)

            if tid not in frame_aparicion:
                frame_aparicion[tid] = frame_count

            # Herencia (solo TIDs nuevos ≤5 frames)
            es_nuevo = (frame_count - frame_aparicion[tid]) <= 5
            if tid not in numero_confirmado and es_nuevo:
                mejor_p = None
                mejor_d = DIST_HERENCIA
                for tid_p, dp in list(jugadores_perdidos.items()):
                    if frame_count - dp['frame'] > FRAMES_HERENCIA:
                        jugadores_perdidos.pop(tid_p, None)
                        continue
                    d = np.hypot(centro_x - dp['cx'], cy_box - dp['cy'])
                    if d < mejor_d:
                        mejor_d = d; mejor_p = tid_p
                if mejor_p is not None:
                    dp = jugadores_perdidos.pop(mejor_p)
                    numero_confirmado[tid] = dp['numero']
                    votos.pop(mejor_p, None)

            y_central  = y_central_en_x(centro_x)
            diferencia = y2 - y_central

            # Color torso (EMA)
            if ancho > 40 and alto > 80:
                tx1 = x1 + int(ancho * 0.2); tx2 = x2 - int(ancho * 0.2)
                ty1 = y1 + int(alto * 0.15); ty2 = y1 + int(alto * 0.45)
                torso = frame[ty1:ty2, tx1:tx2]
                if torso.size > 0:
                    c_act = np.mean(torso.reshape(-1, 3), axis=0).astype(float)
                    color_jugador[tid] = (c_act if tid not in color_jugador
                                          else 0.85 * color_jugador[tid] + 0.15 * c_act)
                    if diferencia > MARGEN_AMBIGUO:
                        colores_cercano.append(color_jugador[tid].copy())
                    elif diferencia < -MARGEN_AMBIGUO:
                        colores_lejano.append(color_jugador[tid].copy())

            # Cercano / lejano con histéresis
            if abs(diferencia) > MARGEN_AMBIGUO:
                es_cercano_raw = diferencia > 0
            elif centroide_cercano is not None and centroide_lejano is not None and tid in color_jugador:
                dc = np.linalg.norm(color_jugador[tid] - centroide_cercano)
                dl = np.linalg.norm(color_jugador[tid] - centroide_lejano)
                es_cercano_raw = dc < dl
            else:
                es_cercano_raw = diferencia > 0

            lado_historial.setdefault(tid, deque(maxlen=15))
            lado_historial[tid].append(es_cercano_raw)
            es_cercano = sum(lado_historial[tid]) > len(lado_historial[tid]) / 2
            es_cercano_dict[tid] = es_cercano

            if not es_cercano:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue

            # OCR
            ocr_listo = (tid not in numero_confirmado
                         and frame_count - ultimo_ocr_frame.get(tid, -OCR_COOLDOWN) >= OCR_COOLDOWN
                         and ancho > 40 and alto > 80)
            if ocr_listo:
                mx  = int(ancho * 0.05)
                x1r = max(0, x1 - mx)
                x2r = min(ancho_frame, x2 + mx)
                rec = frame[y1 + int(alto * 0.05):y1 + int(alto * 0.65), x1r:x2r]
                if rec.size > 0:
                    ultimo_ocr_frame[tid] = frame_count
                    escala  = max(3, 200 // rec.shape[0])
                    rec_g   = cv2.resize(rec, (rec.shape[1] * escala, rec.shape[0] * escala),
                                         interpolation=cv2.INTER_CUBIC)
                    h_g, w_g = rec_g.shape[:2]
                    gris    = cv2.cvtColor(rec_g, cv2.COLOR_BGR2GRAY)
                    gris_cl = clahe.apply(gris)
                    mejor_crop = None
                    for tflag in [cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
                        _, bin_ = cv2.threshold(gris_cl, 0, 255, tflag)
                        ker  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                        bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, ker)
                        n, _, stats_, _ = cv2.connectedComponentsWithStats(bin_)
                        digs = []
                        for i in range(1, n):
                            cx_, cy_, cw_, ch_, ca_ = stats_[i]
                            if cw_ == 0: continue
                            asp = ch_ / cw_
                            rel = ca_ / (h_g * w_g)
                            if (1.0 < asp < 4.5 and 0.02 < rel < 0.45
                                    and w_g * 0.05 < cx_ + cw_ / 2 < w_g * 0.95):
                                digs.append((cx_, cy_, cw_, ch_, ca_))
                        if not digs: continue
                        digs.sort(key=lambda d: d[4], reverse=True)
                        sel = sorted(digs[:2], key=lambda d: d[0])
                        xm = min(d[0] for d in sel); ym = min(d[1] for d in sel)
                        xM = max(d[0] + d[2] for d in sel)
                        yM = max(d[1] + d[3] for d in sel)
                        pad = int((yM - ym) * 0.25)
                        c = rec_g[max(0, ym - pad):min(h_g, yM + pad),
                                  max(0, xm - pad):min(w_g, xM + pad)]
                        if c.size > 0:
                            mejor_crop = c; break
                    imgs_ocr = []
                    if mejor_crop is not None:
                        mc_g = cv2.cvtColor(mejor_crop, cv2.COLOR_BGR2GRAY)
                        imgs_ocr.append(cv2.bitwise_not(mc_g))
                    imgs_ocr.append(cv2.bitwise_not(gris_cl))
                    with _ocr_cond:
                        _pending[tid] = imgs_ocr
                        _ocr_cond.notify()

            numero = numero_confirmado.get(tid, "")
            color  = (0, 255, 0) if numero else (0, 150, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"#{numero}" if numero else f"ID:{tid}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Jugadores que desaparecieron
    for tid_p in ids_prev_frame - ids_este_frame:
        if (tid_p in numero_confirmado and tid_p not in jugadores_perdidos
                and votos.get(tid_p)):
            pos = ultima_pos.get(tid_p, (ancho_frame // 2, altura_frame // 2))
            jugadores_perdidos[tid_p] = {
                'numero': numero_confirmado[tid_p],
                'frame':  frame_count,
                'cx': pos[0], 'cy': pos[1],
            }
    ids_prev_frame = ids_este_frame

    # Líneas de cancha
    if ya_izq is not None:
        cv2.line(annotated, (0, ya_izq), (ancho_frame, ya_der), (255, 200, 0), 1)
    if yr_izq is not None:
        cv2.line(annotated, (0, yr_izq), (ancho_frame, yr_der), (0, 255, 255), 2)
    if y_central_izq is not None:
        cv2.line(annotated, (0, y_central_izq), (ancho_frame, y_central_der), (0, 255, 0), 2)

    # Pelota
    if bola_pos is not None:
        cv2.circle(annotated, bola_pos, 18, (0, 255, 255), 2)

    cv2.putText(annotated,
                f"red:{ratio_ataque_red:.2f}(+/-)  central:{ratio_central_piso:.2f}([/])",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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
