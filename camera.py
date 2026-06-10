import cv2
import numpy as np
import os

# 🔥 Fire detection
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 50, 50])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    fire_pixels = cv2.countNonZero(mask)

    return fire_pixels > 15000   # 🔥 threshold increased


# 🔪 Weapon detection (basic demo logic)
def detect_weapon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    score = cv2.countNonZero(edges)

    return score > 40000   # 🔪 threshold increased


# 🔊 Alarm sound
def play_alarm():
    os.startfile("C:\\Windows\\Media\\Alarm01.wav")


# 🎥 Camera start
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    fire = detect_fire(frame)
    weapon = detect_weapon(frame)

    # 🔥 Fire alert
    if fire:
        cv2.putText(frame, "FIRE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        play_alarm()

    # 🔪 Weapon alert (ONLY if fire not detected)
    elif weapon:
        cv2.putText(frame, "WEAPON DETECTED!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)
        play_alarm()

    # 📺 Show camera
    cv2.imshow("Campus Guardian", frame)

    # ❌ Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()