import cv2
import numpy as np
from ultralytics import YOLO
import winsound
import threading
import time

# 🔥 Load YOLO model
model = YOLO("yolov8n.pt")

# 🔊 Alarm
def play_alarm():
    winsound.PlaySound("C:\\Windows\\Media\\Alarm01.wav", winsound.SND_ASYNC)

def trigger_alarm():
    threading.Thread(target=play_alarm).start()

# 🔥 Fire detection (HSV)
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 50, 50])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    return cv2.countNonZero(mask) > 15000

# 👊 Fight detection (motion-based)
def detect_fight(frame, prev_frame):
    if prev_frame is None:
        return False

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion = cv2.countNonZero(thresh)

    return motion > 60000   # adjust if needed

# ⏱ Cooldown system
last_alert_time = 0
cooldown = 3

# 🧠 Previous frame (for fight detection)
prev_frame = None

# 🎥 Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    fire = detect_fire(frame)
    fight = detect_fight(frame, prev_frame)
    prev_frame = frame.copy()

    # 🔥 PRIORITY 1: FIRE
    if fire and current_time - last_alert_time > cooldown:
        cv2.putText(frame, "FIRE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        trigger_alarm()
        last_alert_time = current_time

    # 👊 PRIORITY 2: FIGHT
    elif fight and current_time - last_alert_time > cooldown:
        cv2.putText(frame, "FIGHT DETECTED!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        trigger_alarm()
        last_alert_time = current_time

    else:
        # 🔪 PRIORITY 3: WEAPON (YOLO)
        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]

                if name in ["knife", "scissors"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 255), 2)

                    cv2.putText(frame, "WEAPON DETECTED!",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2)

                    if current_time - last_alert_time > cooldown:
                        trigger_alarm()
                        last_alert_time = current_time

    cv2.imshow("Campus Guardian PRO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()