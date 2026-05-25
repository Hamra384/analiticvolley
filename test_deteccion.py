import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

source = "D:/AIVolley/analiticvolley/videos/partido.mp4"

cap = cv2.VideoCapture(source)

minutos = 30
segundos = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frame_inicio = int((minutos * 60 + segundos) * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

cv2.namedWindow("Tracking de jugadores", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tracking de jugadores", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        classes=[0],
        persist=True,
        tracker="bytetrack_custom.yaml",
        imgsz=960,
        conf=0.10,
        verbose=False
    )

    annotated = results[0].plot(
        line_width=1,
        font_size=0.4,
        conf=False
    )

    cv2.imshow("Tracking de jugadores", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()