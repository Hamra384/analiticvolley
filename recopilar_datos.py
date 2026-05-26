"""
Recolecta crops de dorsales etiquetados automáticamente desde uno o más videos.
Solo guarda cuando EasyOCR tiene confianza >= MIN_CONF (detecciones seguras).

Uso:
    python recopilar_datos.py                        # usa videos/partido.mp4
    python recopilar_datos.py videos/partido2.mp4
    python recopilar_datos.py videos/p1.mp4 videos/p2.mp4

Salida: training_data/{numero}/*.jpg
Siguiente paso: python entrenar_dorsal.py
"""
import cv2, sys, os, easyocr, numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────
MIN_CONF     = 0.85  # confianza mínima para aceptar etiqueta automática
SKIP         = 30    # procesar 1 de cada N frames (~1 frame/seg a 30fps)
MAX_MINUTOS  = 15    # procesar solo los primeros N minutos por video
META_X_NUM   = 150   # parar cuando todos los números vistos tengan >= este valor
OUTPUT_DIR   = 'training_data'
# ───────────────────────────────────────────────────────────────────────────

videos = sys.argv[1:] if len(sys.argv) > 1 else ['videos/partido.mp4']

yolo   = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'], gpu=True)
clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
os.makedirs(OUTPUT_DIR, exist_ok=True)

conteo = defaultdict(int)

for video_path in videos:
    if not os.path.exists(video_path):
        print(f"[!] No encontrado: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(MAX_MINUTOS * 60 * fps)
    print(f"\n{video_path}  ({total} frames, {fps:.1f} fps)")
    print(f"  Procesando primeros {MAX_MINUTOS} min ({max_frames} frames), 1 cada {SKIP}")

    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_n >= max_frames:
            break
        frame_n += 1
        if frame_n % SKIP != 0:
            continue

        # Parar si ya tenemos suficientes datos de todos los números vistos
        if conteo and min(conteo.values()) >= META_X_NUM:
            print(f"  Meta alcanzada ({META_X_NUM} crops/número)")
            break

        procesados = frame_n // SKIP
        if procesados % 50 == 0:
            print(f"  frame {frame_n}/{max_frames}  |  {dict(sorted(conteo.items()))}")

        results = yolo.track(frame, classes=[0], persist=True,
                             tracker='bytetrack_custom.yaml',
                             imgsz=640, conf=0.25, verbose=False)
        if results[0].boxes is None or results[0].boxes.id is None:
            continue

        h_f, w_f = frame.shape[:2]
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy()

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            alto  = y2 - y1
            ancho = x2 - x1
            if ancho < 40 or alto < 80:
                continue

            mx  = int(ancho * 0.05)
            x1r = max(0, x1 - mx); x2r = min(w_f, x2 + mx)
            rec = frame[y1 + int(alto*0.05): y1 + int(alto*0.65), x1r:x2r]
            if rec.size == 0:
                continue

            escala = max(3, 200 // rec.shape[0])
            rec_g  = cv2.resize(rec, (rec.shape[1]*escala, rec.shape[0]*escala),
                                interpolation=cv2.INTER_CUBIC)
            h_g, w_g = rec_g.shape[:2]
            gris     = cv2.cvtColor(rec_g, cv2.COLOR_BGR2GRAY)
            gris_cl  = clahe.apply(gris)

            # CC para aislar la región del número
            mejor_crop = None
            for tflag in [cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
                _, bin_ = cv2.threshold(gris_cl, 0, 255, tflag)
                ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, ker)
                n, _, stats, _ = cv2.connectedComponentsWithStats(bin_)
                digs = []
                for i in range(1, n):
                    cx, cy, cw, ch, ca = stats[i]
                    if cw == 0: continue
                    asp = ch / cw; rel = ca / (h_g * w_g)
                    if (1.0 < asp < 4.5 and 0.02 < rel < 0.45
                            and w_g*0.05 < cx+cw/2 < w_g*0.95):
                        digs.append((cx, cy, cw, ch, ca))
                if not digs: continue
                digs.sort(key=lambda d: d[4], reverse=True)
                sel = sorted(digs[:2], key=lambda d: d[0])
                xm = min(d[0] for d in sel); ym = min(d[1] for d in sel)
                xM = max(d[0]+d[2] for d in sel); yM = max(d[1]+d[3] for d in sel)
                pad = int((yM-ym)*0.25)
                c = rec_g[max(0,ym-pad):min(h_g,yM+pad), max(0,xm-pad):min(w_g,xM+pad)]
                if c.size > 0:
                    mejor_crop = c; break

            imgs = []
            if mejor_crop is not None:
                mc_g = cv2.cvtColor(mejor_crop, cv2.COLOR_BGR2GRAY)
                imgs += [mejor_crop, mc_g, cv2.bitwise_not(mc_g)]
            imgs += [rec_g, cv2.bitwise_not(gris_cl)]

            candidatos = []
            for img in imgs:
                r = reader.readtext(img, allowlist='0123456789', min_size=6,
                                    text_threshold=0.3, low_text=0.15, width_ths=0.9)
                candidatos.extend(r)

            if not candidatos:
                continue
            mejor = max(candidatos, key=lambda x: x[2])
            texto = mejor[1].strip()
            conf  = mejor[2]

            if texto.isdigit() and 1 <= len(texto) <= 2 and conf >= MIN_CONF:
                carpeta = os.path.join(OUTPUT_DIR, texto)
                os.makedirs(carpeta, exist_ok=True)
                idx = conteo[texto]
                # Guardar el torso completo (más contexto para el modelo)
                cv2.imwrite(f'{carpeta}/{idx:05d}.jpg', rec_g)
                conteo[texto] += 1

    cap.release()

print("\n=== Recolección completada ===")
for num in sorted(conteo, key=int):
    print(f"  #{num}: {conteo[num]} crops")
print(f"\nTotal: {sum(conteo.values())} crops en '{OUTPUT_DIR}/'")
if sum(conteo.values()) < 50:
    print("[!] Pocos datos — probá con más videos o bajá MIN_CONF a 0.75")
else:
    print("Siguiente paso: python entrenar_dorsal.py")
