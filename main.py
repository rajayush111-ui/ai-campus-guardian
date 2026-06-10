from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import time
import winsound

app = FastAPI()

# ─── CORS — allows the frontend HTML file to call this backend ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to ["http://127.0.0.1:5500"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 Load YOLO model
model = YOLO("yolov8n.pt")

# 🎥 Camera
camera = cv2.VideoCapture(0)

# ⏱ Cooldown (kept for /test-sound endpoint compatibility)
last_alert_time = 0
cooldown = 2

# 👊 Previous frame
prev_frame = None

# ─── ALARM STATE — prevents continuous beeping ───
# alert_active  : True while a threat is ongoing; alarm won't re-fire until
#                 the threat CLEARS and a NEW threat appears.
# *_confirm     : each detection function must return True for CONFIRM_FRAMES
#                 consecutive frames before the alarm fires — filters out
#                 single-frame noise, compression glitches, and micro-flickers.
alert_active   = False
fire_confirm   = 0
fight_confirm  = 0
weapon_confirm = 0
CONFIRM_FRAMES = 4       # ~4 frames at ~15fps ≈ 0.25s of sustained detection


# 🔊 ALERT SOUNDS — run in daemon threads so they don't block the frame loop.
# Blocking winsound.Beep() was stalling frame generation and causing
# the next frame to be processed late, sometimes double-triggering.
import threading

def _beep(freq, dur):
    winsound.Beep(freq, dur)

def fire_alarm():
    threading.Thread(target=_beep, args=(1000, 400), daemon=True).start()

def fight_alarm():
    threading.Thread(target=_beep, args=(700,  400), daemon=True).start()

def weapon_alarm():
    threading.Thread(target=_beep, args=(500,  400), daemon=True).start()


# 🔥 Fire detection
# Two-band HSV mask: true fire has BOTH high saturation AND high brightness.
# Skin, walls, and warm lighting are filtered out by requiring sat > 150
# and by cross-checking with a brightness mask (val > 200).
# Pixel threshold raised to 8000 so a small warm patch doesn't trigger.
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Primary orange-red fire range with tightened saturation floor
    lower = np.array([0,  150, 200])   # sat≥150, val≥200 → vivid bright orange only
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological open: removes isolated noise pixels (salt/pepper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    fire_pixels = cv2.countNonZero(mask)

    # Must be a meaningful contiguous blob, not scattered pixels
    return 8000 < fire_pixels < 60000


# 👊 Fight detection
# motion > 20000 was far too sensitive — normal walking, hand movement,
# even autofocus flicker triggers it. Raised to 80000 which requires
# large fast movement across a significant portion of the frame.
# Also added a blur pre-step to suppress camera noise / compression artifacts.
def detect_fight(frame, prev_frame):
    if prev_frame is None:
        return False

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY)

    # Gaussian blur removes compression noise before diffing
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion = cv2.countNonZero(thresh)

    return motion > 80000   # raised from 20000 → requires large rapid movement


@app.get("/")
def home():
    return {"message": "Campus Guardian Running"}


@app.get("/camera")
def camera_status():
    return {"status": "Camera initialized"}


# 🎥 Frame generator
def generate_frames():
    global prev_frame
    global alert_active, fire_confirm, fight_confirm, weapon_confirm

    while True:
        success, frame = camera.read()
        if not success:
            break

        # ── Per-frame detection ──────────────────────────────────────────
        fire  = detect_fire(frame)
        fight = detect_fight(frame, prev_frame)
        prev_frame = frame.copy()

        detected_this_frame = False   # True if ANY real threat seen this frame

        # ── 🔥 FIRE ─────────────────────────────────────────────────────
        if fire:
            fire_confirm += 1
            cv2.putText(frame, "FIRE DETECTED!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            detected_this_frame = True

            # Alarm fires only when confirmed AND not already active
            if fire_confirm >= CONFIRM_FRAMES and not alert_active:
                fire_alarm()
                alert_active = True
        else:
            fire_confirm = 0   # reset — detection must be consecutive

        # ── 👊 FIGHT ─────────────────────────────────────────────────────
        if fight and not fire:   # fight only if no fire (priority unchanged)
            fight_confirm += 1
            cv2.putText(frame, "FIGHT DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            detected_this_frame = True

            if fight_confirm >= CONFIRM_FRAMES and not alert_active:
                fight_alarm()
                alert_active = True
        else:
            fight_confirm = 0

        # ── 🔪 WEAPON (YOLO — only when no fire/fight) ───────────────────
        if not detected_this_frame:
            results = model(frame)
            weapon_seen = False

            for r in results:
                for box in r.boxes:
                    cls  = int(box.cls[0])
                    name = model.names[cls]

                    if name in ["knife", "scissors", "bottle", "cell phone"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 255, 255), 2)
                        cv2.putText(frame, "WEAPON DETECTED!",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 255), 2)
                        weapon_seen        = True
                        detected_this_frame = True

            if weapon_seen:
                weapon_confirm += 1
                if weapon_confirm >= CONFIRM_FRAMES and not alert_active:
                    weapon_alarm()
                    alert_active = True
            else:
                weapon_confirm = 0

        # ── Reset alert_active when threat fully clears ───────────────────
        # All three confirm counters must be zero (no detection in progress)
        # before we allow the alarm to fire again for a new event.
        if not detected_this_frame:
            alert_active = False

        # ── Encode and stream frame ───────────────────────────────────────
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

import os

@app.get("/test-sound")
def test():
    sound_path = "C:\\Windows\\Media\\Alarm01.wav"
    os.startfile(sound_path)
    return {"status": "sound triggered"}


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
