import cv2
import numpy as np
import threading
import json
import os
from collections import deque
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# ── Selección de motor OCR ────────────────────────────────────────────────────
# Si existe dorsal_classifier.onnx (entrenado con recopilar_datos + entrenar_dorsal)
# lo usa inline (~2ms/jugador). Si no, cae a EasyOCR con worker asíncrono.
_USE_DORSAL = (os.path.exists('dorsal_classifier.onnx')
               and os.path.exists('dorsal_classes.json'))

if _USE_DORSAL:
    import onnxruntime as ort
    _ort   = ort.InferenceSession('dorsal_classifier.onnx',
                                   providers=['CUDAExecutionProvider',
                                              'CPUExecutionProvider'])
    _clases = json.load(open('dorsal_classes.json'))
    _MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    print(f"[dorsal] modelo cargado — clases: {_clases}")

    def _ocr_crop(recorte_bgr):
        img = cv2.resize(recorte_bgr, (96, 96)).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        inp = img.transpose(2, 0, 1)[np.newaxis]
        logits = _ort.run(None, {'input': inp})[0][0]
        probs  = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx    = int(np.argmax(probs))
        return _clases[idx], float(probs[idx])

else:
    import easyocr
    reader    = easyocr.Reader(['en'], gpu=True)
    _pending  = {}
    _results  = {}
    _ocr_cond = threading.Condition()
    print("[EasyOCR] modelo dorsal no encontrado — usando EasyOCR asíncrono")
    print("  Para mejor velocidad: python recopilar_datos.py && python entrenar_dorsal.py")

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

source = "D:/AIVolley/analiticvolley/videos/partido.mp4"

cap = cv2.VideoCapture(source)

minutos = 14
segundos = 0
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

# Detector de pelota: background subtraction + HoughCircles
bg_sub      = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)
_kern_ball  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

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
ultimo_ocr_frame = {}   # tid -> último frame en que se encoló OCR
OCR_COOLDOWN = 3        # frames mínimos entre intentos OCR por jugador
es_cercano_dict = {}    # tid -> bool, actualizado cada frame

# ── Detección de cluster (jugadores agrupados) ────────────────────────────
CLUSTER_DIST    = 80   # px: distancia máxima centro-centro para considerar "juntos"
CLUSTER_MIN     = 5    # jugadores "juntos" necesarios para triggerear
FRAMES_ESPERA   = 45   # 1.5 s a 30 fps antes de re-trackear
FRAMES_COOLDOWN = 150  # 5 s sin poder re-triggerear tras resolver un cluster
en_cluster          = False
cluster_inicio      = 0
cluster_cooldown    = 0  # frame hasta el que no se puede re-triggerear
tids_en_cluster     = set()

# ── Pelota y estadísticas de punto ───────────────────────────────────────────
bola_hist           = deque(maxlen=12)   # posiciones crudas (None si no detectada)
bola_pos            = None               # posición suavizada (cx, cy)
bola_vel            = 0.0               # px/frame
bola_frames_quieta  = 0                 # frames consecutivos sin movimiento

UMBRAL_VEL_INICIO   = 12   # px/frame mínimo para iniciar punto
UMBRAL_VEL_QUIETA   = 5    # px/frame máximo para considerar pelota quieta
UMBRAL_BOLA_QUIETA  = 50   # frames quieta = fin de punto (~1.7 s a 30 fps)
DIST_CONTACTO       = 160  # px máx jugador-pelota para registrar contacto
CONTACTO_COOLDOWN   = 8    # frames mínimos entre dos contactos registrados

en_punto            = False
punto_n             = 0
punto_inicio_f      = 0
punto_actual        = {}
historial_puntos    = []
ultimo_contacto_f   = 0


def _snapshot_jugadores():
    """Devuelve dict tid -> {cx, cy, numero} de todos los jugadores cercanos."""
    return {tid: {'cx': p[0], 'cy': p[1],
                  'numero': numero_confirmado.get(tid, '?')}
            for tid, p in ultima_pos.items()
            if es_cercano_dict.get(tid)}


def _guardar_estadisticas():
    import json as _json
    with open('estadisticas.json', 'w', encoding='utf-8') as _f:
        _json.dump(historial_puntos, _f, ensure_ascii=False, indent=2)


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

    # ── Consumir resultados del worker EasyOCR (solo si no hay modelo dorsal) ──
    if not _USE_DORSAL:
        with _ocr_cond:
            res_frame = dict(_results)
            _results.clear()
    else:
        res_frame = {}
    for tid_r, (texto, confianza) in res_frame.items():
        if tid_r in numero_confirmado:
            continue
        numeros_activos = {n for t, n in numero_confirmado.items() if t != tid_r}
        if (texto.isdigit() and 1 <= len(texto) <= 2
                and confianza > 0.38 and texto not in numeros_activos):
            if confianza >= 0.88:
                numero_confirmado[tid_r] = texto
                print(f"ID {tid_r} → #{texto} ({confianza:.2f})")
            else:
                if tid_r not in votos:
                    votos[tid_r] = {}
                votos[tid_r][texto] = votos[tid_r].get(texto, 0) + 1
                votos_nec = 3 if len(texto) == 2 else 2
                if votos[tid_r][texto] >= votos_nec:
                    numero_confirmado[tid_r] = texto
                    print(f"ID {tid_r} → #{texto} ({votos[tid_r][texto]} votos)")

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

    # ── Detección de pelota: background subtraction + HoughCircles ──────────
    gray_b  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_sub.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  _kern_ball)
    fg_mask = cv2.dilate(fg_mask, _kern_ball, iterations=2)

    blurred = cv2.GaussianBlur(gray_b, (7, 7), 1.5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=40,
                               param1=50, param2=18,
                               minRadius=6, maxRadius=30)
    bola_cruda = None
    if circles is not None:
        player_boxes = (results[0].boxes.xyxy.cpu().numpy().astype(int)
                        if results[0].boxes is not None else [])
        mejor_mov = 0
        for cx_c, cy_c, r_c in circles[0]:
            cx_c, cy_c, r_c = int(cx_c), int(cy_c), int(r_c)
            if cy_c < altura_frame * 0.08:          # descartar marcador/techo
                continue
            x1b = max(0, cx_c - r_c); x2b = min(ancho_frame, cx_c + r_c)
            y1b = max(0, cy_c - r_c); y2b = min(altura_frame, cy_c + r_c)
            mov = float(fg_mask[y1b:y2b, x1b:x2b].mean()) if (x2b > x1b and y2b > y1b) else 0
            if mov < 15:                             # necesita movimiento real
                continue
            en_jugador = any(int(b[0]) < cx_c < int(b[2]) and int(b[1]) < cy_c < int(b[3])
                             for b in player_boxes)
            if en_jugador:                           # ignorar blobs dentro de jugadores
                continue
            if mov > mejor_mov:
                mejor_mov = mov
                bola_cruda = (cx_c, cy_c)

    bola_hist.append(bola_cruda)
    pos_validas = [p for p in bola_hist if p is not None]
    if len(pos_validas) >= 2:
        bola_pos = (int(np.mean([p[0] for p in pos_validas[-4:]])),
                    int(np.mean([p[1] for p in pos_validas[-4:]])))
        bola_vel = float(np.hypot(pos_validas[-1][0] - pos_validas[-2][0],
                                  pos_validas[-1][1] - pos_validas[-2][1]))
    elif pos_validas:
        bola_pos = pos_validas[-1]
        bola_vel = 0.0
    else:
        bola_pos = None
        bola_vel = 0.0

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

            es_cercano_dict[tid] = es_cercano
            if not es_cercano:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue

            # Durante un cluster activo: solo dibujar, no OCR
            if en_cluster:
                numero = numero_confirmado.get(tid, "")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(annotated, f"ID:{tid}" if not numero else f"#{numero}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                continue

            ocr_listo = (tid not in numero_confirmado
                         and frame_count - ultimo_ocr_frame.get(tid, -OCR_COOLDOWN) >= OCR_COOLDOWN
                         and ancho > 40 and alto > 80)
            if ocr_listo:
                margen_x = int(ancho * 0.05)
                x1r = max(0, x1 - margen_x)
                x2r = min(ancho_frame, x2 + margen_x)
                recorte = frame[y1 + int(alto * 0.05):y1 + int(alto * 0.65), x1r:x2r]

                if recorte.size > 0:
                    ultimo_ocr_frame[tid] = frame_count

                    if _USE_DORSAL:
                        # Modelo entrenado: inline, ~2ms
                        texto, confianza = _ocr_crop(recorte)
                        numeros_activos = {n for t, n in numero_confirmado.items() if t != tid}
                        if texto not in numeros_activos and confianza > 0.6:
                            if confianza >= 0.90:
                                numero_confirmado[tid] = texto
                                print(f"ID {tid} → #{texto} ({confianza:.2f})")
                            else:
                                if tid not in votos:
                                    votos[tid] = {}
                                votos[tid][texto] = votos[tid].get(texto, 0) + 1
                                votos_nec = 3 if len(texto) == 2 else 2
                                if votos[tid][texto] >= votos_nec:
                                    numero_confirmado[tid] = texto
                                    print(f"ID {tid} → #{texto} ({votos[tid][texto]} votos)")
                    else:
                        # EasyOCR: preprocessing + worker asíncrono
                        escala = max(3, 200 // recorte.shape[0])
                        recorte_g = cv2.resize(recorte,
                                               (recorte.shape[1]*escala, recorte.shape[0]*escala),
                                               interpolation=cv2.INTER_CUBIC)
                        h_g, w_g = recorte_g.shape[:2]
                        gris = cv2.cvtColor(recorte_g, cv2.COLOR_BGR2GRAY)
                        gris_clahe = clahe.apply(gris)

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
                                if (1.0 < aspecto < 4.5 and 0.02 < rel < 0.45
                                        and w_g*0.05 < cx+cw/2 < w_g*0.95):
                                    digitos.append((cx, cy, cw, ch, ca))
                            if not digitos:
                                continue
                            digitos.sort(key=lambda d: d[4], reverse=True)
                            sel = sorted(digitos[:2], key=lambda d: d[0])
                            xm = min(d[0] for d in sel); ym = min(d[1] for d in sel)
                            xM = max(d[0]+d[2] for d in sel); yM = max(d[1]+d[3] for d in sel)
                            pad = int((yM-ym)*0.25)
                            c = recorte_g[max(0,ym-pad):min(h_g,yM+pad),
                                          max(0,xm-pad):min(w_g,xM+pad)]
                            if c.size > 0:
                                mejor_crop = c
                                break

                        imgs_ocr = []
                        if mejor_crop is not None:
                            mc_g = cv2.cvtColor(mejor_crop, cv2.COLOR_BGR2GRAY)
                            imgs_ocr.append(cv2.bitwise_not(mc_g))
                        imgs_ocr.append(cv2.bitwise_not(gris_clahe))

                        with _ocr_cond:
                            _pending[tid] = imgs_ocr
                            _ocr_cond.notify()

            numero = numero_confirmado.get(tid, "")
            color = (0, 255, 0) if numero else (0, 150, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{numero}" if numero else f"ID:{tid}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Detección y manejo de cluster ────────────────────────────────────────
    pos_cercanos = [(tid, ultima_pos[tid])
                    for tid in ids_este_frame
                    if es_cercano_dict.get(tid) and tid in ultima_pos]

    if not en_cluster and frame_count > cluster_cooldown:
        # Contar cuántos jugadores están "juntos"
        bunched = set()
        for i in range(len(pos_cercanos)):
            for j in range(i + 1, len(pos_cercanos)):
                tid_a, (cx_a, cy_a) = pos_cercanos[i]
                tid_b, (cx_b, cy_b) = pos_cercanos[j]
                if np.hypot(cx_a - cx_b, cy_a - cy_b) < CLUSTER_DIST:
                    bunched.add(tid_a)
                    bunched.add(tid_b)
        if len(bunched) >= CLUSTER_MIN:
            en_cluster      = True
            cluster_inicio  = frame_count
            tids_en_cluster = set(bunched)
            print(f"[cluster] detectado en frame {frame_count} — TIDs: {tids_en_cluster}")
    else:
        if frame_count - cluster_inicio >= FRAMES_ESPERA:
            # Limpiar números de los jugadores que estaban en el cluster
            for tid in tids_en_cluster:
                numero_confirmado.pop(tid, None)
                votos.pop(tid, None)
            jugadores_perdidos.clear()
            en_cluster       = False
            tids_en_cluster  = set()
            cluster_cooldown = frame_count + FRAMES_COOLDOWN
            print(f"[cluster] resuelto en frame {frame_count} — re-identificando (cooldown hasta frame {cluster_cooldown})")
        else:
            restantes = FRAMES_ESPERA - (frame_count - cluster_inicio)
            cv2.putText(annotated, f"AGRUPADOS — esperando {restantes}f",
                        (20, altura_frame - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 165, 255), 2)

    # ── Lógica de inicio / fin de punto ──────────────────────────────────────
    if bola_vel < UMBRAL_VEL_QUIETA or bola_pos is None:
        bola_frames_quieta += 1
    else:
        bola_frames_quieta = 0

    if not en_punto:
        if bola_vel >= UMBRAL_VEL_INICIO and bola_pos is not None:
            en_punto      = True
            punto_n      += 1
            punto_inicio_f = frame_count
            # Sacador: jugador cercano más próximo a la pelota al momento del saque
            sacador_tid  = None
            mejor_d      = float('inf')
            for tid, pos in ultima_pos.items():
                if not es_cercano_dict.get(tid):
                    continue
                d = np.hypot(pos[0] - bola_pos[0], pos[1] - bola_pos[1])
                if d < mejor_d:
                    mejor_d = d; sacador_tid = tid
            num_sac = numero_confirmado.get(sacador_tid, f'ID:{sacador_tid}') if sacador_tid else '?'
            punto_actual = {
                'numero': punto_n,
                'frame_inicio': frame_count,
                'frame_fin':    None,
                'duracion_s':   None,
                'pos_inicio':   _snapshot_jugadores(),
                'pos_fin':      {},
                'sacador':      num_sac,
                'sacador_pos':  ultima_pos.get(sacador_tid),
                'armador':      None, 'armador_pos':   None,
                'rematador':    None, 'rematador_pos': None,
                'bloqueadores': [],
                'defensores':   [],
                'contactos':    [],
            }
            print(f"[PUNTO {punto_n}] INICIO frame {frame_count} — saque #{num_sac}")
    else:
        # ── Detectar contacto (cambio brusco de dirección) ──────────────────
        if (bola_pos is not None and len(pos_validas) >= 3
                and frame_count - ultimo_contacto_f >= CONTACTO_COOLDOWN):
            vxp = pos_validas[-2][0] - pos_validas[-3][0]
            vyp = pos_validas[-2][1] - pos_validas[-3][1]
            vxc = pos_validas[-1][0] - pos_validas[-2][0]
            vyc = pos_validas[-1][1] - pos_validas[-2][1]
            mag_p = np.hypot(vxp, vyp)
            mag_c = np.hypot(vxc, vyc)
            if mag_p > 4 and mag_c > 4:
                cos_a = (vxp*vxc + vyp*vyc) / (mag_p * mag_c)
                if cos_a < 0.4:   # cambio de dirección ≥ ~66°
                    jugadores_cerca = [(t, p) for t, p in ultima_pos.items()
                                       if es_cercano_dict.get(t)]
                    if jugadores_cerca:
                        mejor = min(jugadores_cerca,
                                    key=lambda x: np.hypot(x[1][0]-bola_pos[0],
                                                            x[1][1]-bola_pos[1]))
                        dist_bola = np.hypot(mejor[1][0]-bola_pos[0], mejor[1][1]-bola_pos[1])
                        if dist_bola < DIST_CONTACTO:
                            ultimo_contacto_f = frame_count
                            tid_c  = mejor[0]
                            num_c  = numero_confirmado.get(tid_c, f'ID:{tid_c}')
                            pos_c  = mejor[1]
                            y_red_c = y_red_en_x(pos_c[0])
                            cerca_red = abs(pos_c[1] - y_red_c) < altura_frame * 0.15
                            contacto = {
                                'frame':      frame_count,
                                'numero':     num_c,
                                'pos':        list(pos_c),
                                'pos_bola':   list(bola_pos),
                                'cerca_red':  cerca_red,
                                'vel_salida': round(float(mag_c), 1),
                            }
                            punto_actual['contactos'].append(contacto)
                            # Clasificar acción por posición y velocidad de salida
                            if cerca_red and mag_c > 18:
                                if punto_actual['rematador'] is None:
                                    punto_actual['rematador']     = num_c
                                    punto_actual['rematador_pos'] = list(pos_c)
                            elif mag_c <= 18 and mag_c > 4:
                                if punto_actual['armador'] is None:
                                    punto_actual['armador']     = num_c
                                    punto_actual['armador_pos'] = list(pos_c)
                            elif not cerca_red:
                                if num_c not in punto_actual['defensores']:
                                    punto_actual['defensores'].append(num_c)

        # ── Fin de punto ────────────────────────────────────────────────────
        if bola_frames_quieta >= UMBRAL_BOLA_QUIETA:
            punto_actual['frame_fin']  = frame_count
            punto_actual['duracion_s'] = round((frame_count - punto_inicio_f) / fps, 1)
            punto_actual['pos_fin']    = _snapshot_jugadores()
            # Bloqueadores: contactos cerca de la red con vel de salida baja (pelota bloqueada)
            contactos = punto_actual['contactos']
            for i, c in enumerate(contactos):
                if c['cerca_red'] and c['vel_salida'] < 15:
                    if c['numero'] not in punto_actual['bloqueadores']:
                        punto_actual['bloqueadores'].append(c['numero'])
            en_punto           = False
            bola_frames_quieta = 0
            historial_puntos.append(dict(punto_actual))
            _guardar_estadisticas()
            print(f"[PUNTO {punto_n}] FIN — {punto_actual['duracion_s']}s | "
                  f"saque:{punto_actual['sacador']} "
                  f"arme:{punto_actual['armador']} desde:{punto_actual['armador_pos']} "
                  f"remate:{punto_actual['rematador']} "
                  f"defensa:{punto_actual['defensores']} "
                  f"bloqueo:{punto_actual['bloqueadores']}")

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

    # ── Overlay pelota ───────────────────────────────────────────────────────
    if bola_pos is not None:
        cv2.circle(annotated, bola_pos, 18, (0, 255, 255), 2)
        cv2.putText(annotated, f"v={bola_vel:.0f}", (bola_pos[0] + 14, bola_pos[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # ── Overlay estado del punto ─────────────────────────────────────────────
    if en_punto:
        dur = (frame_count - punto_inicio_f) / fps
        cv2.putText(annotated, f"PUNTO {punto_n}  {dur:.1f}s",
                    (ancho_frame // 2 - 110, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(annotated, f"puntos: {len(historial_puntos)}",
                    (ancho_frame // 2 - 60, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1)

    # ── Último punto completado ──────────────────────────────────────────────
    if historial_puntos:
        u = historial_puntos[-1]
        linea1 = (f"P{u['numero']} ({u['duracion_s']}s)  "
                  f"saque:#{u['sacador']}  "
                  f"arme:#{u['armador']} desde:{u['armador_pos']}  "
                  f"remate:#{u['rematador']}")
        linea2 = (f"defensa:{u['defensores']}  bloqueo:{u['bloqueadores']}")
        cv2.putText(annotated, linea1, (10, altura_frame - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
        cv2.putText(annotated, linea2, (10, altura_frame - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)

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