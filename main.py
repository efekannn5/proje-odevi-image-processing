import cv2
import torch


model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp13/weights/best.pt', force_reload=True)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    results = model(frame)
    rendered = results.render()

    if len(rendered):
        cv2.imshow('EFEKAN NEFESOGLU', rendered[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
