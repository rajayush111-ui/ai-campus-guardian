import cv2
from ultralytics import YOLO
import winsound
import threading

# 🔥 Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight model

# 🔊 Alarm (fast)
def play_alarm():
    winsound.PlaySound("C:\\Windows\\Media\\Alarm01.wav", winsound.SND_ASYNC)

def trigger_alarm():
    threading.Thread(target=play_alarm).start()

# 🎥 Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame)

    detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            # 🔥 Fire / 🔪 Weapon detection
            if name in ["knife", "scissors"]:   # weapon classes
                detected = True
                label = "WEAPON DETECTED!"
                color = (0, 255, 255)

            elif name in ["fire"]:   # (fire default model me nahi hota)
                detected = True
                label = "FIRE DETECTED!"
                color = (0, 0, 255)

            else:
                continue

            # Draw box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if detected:
        trigger_alarm()

    cv2.imshow("AI Campus Guardian", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()