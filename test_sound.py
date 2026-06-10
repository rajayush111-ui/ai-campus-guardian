import os

@app.get("/test-sound")
def test():
    sound_path = "C:\\Windows\\Media\\Alarm01.wav"
    os.startfile(sound_path)
    return {"status": "sound triggered"}