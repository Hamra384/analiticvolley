"""
Diagnóstico OCR: procesa 600 frames (20s), guarda recortes y logs.
Ejecutar: python diag_ocr.py
Salida:   diag_out/  (recortes)  +  diag_ocr.txt  (log)
"""
import os, cv2, easyocr, numpy as np
from ultralytics import YOLO

os.makedirs("diag_out", exist_ok=True)

model  = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=True)
clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
KSHARP = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

cap = cv2.VideoCapture("D:/AIVolley/analiticvolley/videos/partido.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int((14*60 + 10) * fps))

log = open("diag_ocr.txt", "w", encoding="utf-8")

for frame_num in range(600):
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0], persist=True,
                          tracker="bytetrack_custom.yaml",
                          imgsz=640, conf=0.25, verbose=False)

    if results[0].boxes is None or results[0].boxes.id is None:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids   = results[0].boxes.id.cpu().numpy()

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        alto  = y2 - y1
        ancho = x2 - x1
        tid   = int(track_id)

        if ancho < 40 or alto < 80:
            continue

        # ── recorte del torso ──────────────────────────────────────
        mx  = int(ancho * 0.05)
        x1r = max(0, x1 - mx)
        x2r = min(frame.shape[1], x2 + mx)
        rec = frame[y1 + int(alto*0.05): y1 + int(alto*0.65), x1r:x2r]
        if rec.size == 0:
            continue

        escala   = max(3, 200 // rec.shape[0])
        rec_g    = cv2.resize(rec, (rec.shape[1]*escala, rec.shape[0]*escala),
                              interpolation=cv2.INTER_CUBIC)
        gris     = cv2.cvtColor(rec_g, cv2.COLOR_BGR2GRAY)
        sharp    = np.clip(cv2.filter2D(gris, -1, KSHARP), 0, 255).astype(np.uint8)
        gclahe   = clahe.apply(gris)
        inv      = cv2.bitwise_not(gclahe)

        # ── connected components → intentar aislar número ─────────
        mejor_crop = None
        for tflag in [cv2.THRESH_BINARY     + cv2.THRESH_OTSU,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU]:
            _, bin_ = cv2.threshold(gclahe, 0, 255, tflag)
            ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, ker)
            n, _, stats, _ = cv2.connectedComponentsWithStats(bin_)
            h_g, w_g = rec_g.shape[:2]
            area_t = h_g * w_g
            digs = []
            for i in range(1, n):
                cx2,cy2,cw,ch,ca = stats[i]
                if cw == 0: continue
                asp = ch/cw
                rel = ca/area_t
                if 1.0 < asp < 4.5 and 0.02 < rel < 0.45 and w_g*0.05 < cx2+cw/2 < w_g*0.95:
                    digs.append((cx2,cy2,cw,ch,ca))
            if not digs: continue
            digs.sort(key=lambda d: d[4], reverse=True)
            sel = sorted(digs[:2], key=lambda d: d[0])
            xm = min(d[0] for d in sel); ym = min(d[1] for d in sel)
            xM = max(d[0]+d[2] for d in sel); yM = max(d[1]+d[3] for d in sel)
            pad = int((yM-ym)*0.25)
            xm=max(0,xm-pad); ym=max(0,ym-pad)
            xM=min(w_g,xM+pad); yM=min(h_g,yM+pad)
            c = rec_g[ym:yM, xm:xM]
            if c.size > 0:
                mejor_crop = c
                break

        # ── OCR ───────────────────────────────────────────────────
        imgs = []
        if mejor_crop is not None:
            mg = cv2.cvtColor(mejor_crop, cv2.COLOR_BGR2GRAY)
            imgs += [("crop_color", mejor_crop),
                     ("crop_gris",  mg),
                     ("crop_inv",   cv2.bitwise_not(mg))]
        imgs += [("torso_color", rec_g),
                 ("torso_sharp", sharp),
                 ("torso_inv",   inv)]

        dets = []
        for name, img in imgs:
            r = reader.readtext(img, allowlist='0123456789',
                                min_size=6, text_threshold=0.3,
                                low_text=0.15, width_ths=0.9)
            for det in r:
                dets.append(f"{name}:{det[1]}@{det[2]:.2f}")

        line = f"f{frame_num:04d} tid={tid:3d} {ancho}x{alto}  [{', '.join(dets) if dets else 'NADA'}]"
        print(line)
        log.write(line + "\n")

        # Guardar recorte solo si hubo al menos una detección
        if dets:
            cv2.imwrite(f"diag_out/f{frame_num:04d}_tid{tid}.jpg", rec_g)
            if mejor_crop is not None:
                cv2.imwrite(f"diag_out/f{frame_num:04d}_tid{tid}_crop.jpg", mejor_crop)

log.close()
cap.release()
print("\nListo → diag_ocr.txt  y  diag_out/")
