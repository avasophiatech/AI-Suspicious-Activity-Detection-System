from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                if start_time is None:
                    start_time = time.time()

                if time.time() - start_time > 10:
                    cv2.putText(frame, "Suspicious Activity Detected!",
                                (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0,0,255), 3)

    cv2.imshow("Suspicious Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
