from fastapi import FastAPI
import cv2

app = FastAPI()

@app.get("/live")
def live():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Campus Guardian LIVE", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return {"status": "stopped"}