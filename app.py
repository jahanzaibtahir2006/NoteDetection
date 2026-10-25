"""Step 3: run Note Detection live (camera + YOLO + voice + FastAPI dashboard).

    python app.py                                   # laptop webcam
    python app.py --source 1                        # second camera (DroidCam, OBS virtual camera)
    python app.py --source http://192.168.1.5:8080/video   # IP Webcam phone app
    python app.py --source scrcpy                   # phone camera mirrored by scrcpy

Dashboard: http://localhost:8000    API docs: http://localhost:8000/docs
"""
import argparse
from collections import Counter, deque
from pathlib import Path
import re
import threading
import time
import webbrowser

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from assistant import announce, describe, reply
from camera import Camera
from voice import Listener, Speaker

ROOT = Path(__file__).parent
STABLE_FRAMES = 10   # a result must hold for this many frames before it is announced
FORGET_AFTER = 2.0   # seconds without notes before the next note counts as new
VOTES = 15           # recent frames whose class guesses decide a tracked note's value
MIN_HITS = 3         # frames a new note must be seen before it counts
LOST_AFTER = 30      # frames after which a note that left the view is forgotten
GREEN = (88, 157, 15)


def note_value(class_name):
    digits = re.findall(r"\d+", str(class_name))
    return int(digits[0]) if digits else None


class NoteTracker:
    """Gives each note a steady value.

    YOLO decides every frame on its own, so a blurred 1000 note can read as 50 for a
    frame or two. ByteTrack follows each note across frames, and the note's value is
    the class with the highest summed confidence over its last VOTES frames.
    """

    def __init__(self):
        self.votes = {}
        self.last_seen = {}
        self.frame = 0

    def update(self, result):
        """Return [(value, xyxy)] for the notes in this frame."""
        self.frame += 1
        boxes = result.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else []
        notes = []
        for track, cls, conf, xyxy in zip(ids, boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            votes = self.votes.setdefault(track, deque(maxlen=VOTES))
            votes.append((note_value(result.names[int(cls)]), conf))
            self.last_seen[track] = self.frame
            if len(votes) >= MIN_HITS:
                score = Counter()
                for value, weight in votes:
                    score[value] += weight
                notes.append((score.most_common(1)[0][0], xyxy))
        for track in [t for t, seen in self.last_seen.items() if self.frame - seen > LOST_AFTER]:
            del self.votes[track], self.last_seen[track]
        return [(v, b) for v, b in notes if v]


class NoteDetection:
    def __init__(self, weights, conf, imgsz):
        self.model = YOLO(weights)       # live tracking keeps state between frames
        self.api_model = YOLO(weights)   # single images sent to the API
        self.tracker = NoteTracker()
        self.conf = conf
        self.imgsz = imgsz
        self.lock = threading.Lock()
        self.notes = ()
        self.jpeg = None
        self.fps = 0.0
        self.log = deque(maxlen=12)
        self.last_spoken = ""
        self.speaker = None

    def detect(self, frame):
        with self.lock:
            result = self.api_model.predict(frame, conf=self.conf, imgsz=self.imgsz, half=False, verbose=False)[0]
        values = (note_value(result.names[int(c)]) for c in result.boxes.cls)
        return result, sorted((v for v in values if v), reverse=True)

    def track(self, frame):
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf,
                                  imgsz=self.imgsz, half=False, verbose=False)[0]
        return self.tracker.update(result)

    def say(self, text, interrupt=False):
        self.last_spoken = text
        self.log.append({"who": "assistant", "text": text})
        if self.speaker:
            self.speaker.say(text, interrupt)

    def hear(self, text):
        self.log.append({"who": "user", "text": text})
        answer = reply(text, list(self.notes), self.last_spoken)
        if answer:
            self.say(answer)
        return answer

    def run(self, camera):
        history, announced, empty_since = deque(maxlen=STABLE_FRAMES), (), None
        last = time.time()
        while True:
            frame = camera.frame
            if frame is None:
                time.sleep(0.05)
                continue
            tracked = self.track(frame)
            history.append(tuple(sorted((v for v, _ in tracked), reverse=True)))

            # Speak the total once the same set of notes has been seen steadily
            seen, hits = Counter(history).most_common(1)[0]
            if len(history) == STABLE_FRAMES and hits >= STABLE_FRAMES * 0.8:
                self.notes = seen
                if seen:
                    empty_since = None
                    if seen != announced:
                        self.say(announce(list(announced), list(seen)), interrupt=True)
                        announced = seen
                else:
                    # notes briefly leave the frame while being handled; forget them only after a pause
                    empty_since = empty_since or time.time()
                    if time.time() - empty_since > FORGET_AFTER:
                        announced = ()

            image = frame.copy()
            for value, (x1, y1, x2, y2) in tracked:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 3)
                label = f"Rs {value:,}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(image, (x1, max(y1 - h - 14, 0)), (x1 + w + 12, max(y1, h + 14)), GREEN, -1)
                cv2.putText(image, label, (x1 + 6, max(y1 - 8, h + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (image.shape[1], 56), (255, 255, 255), -1)
            cv2.putText(image, f"You have Rs {sum(self.notes):,}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 2, cv2.LINE_AA)
            self.jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now


class Question(BaseModel):
    text: str


class Settings(BaseModel):
    conf: float


def create_api(sense):
    api = FastAPI(title="Note Detection API", description="Real-time Pakistani banknote recognition")

    @api.get("/", response_class=HTMLResponse)
    def dashboard():
        return (ROOT / "dashboard.html").read_text(encoding="utf-8")

    @api.get("/video")
    def video():
        """Live annotated camera feed (MJPEG)."""
        def frames():
            while True:
                if sense.jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + sense.jpeg + b"\r\n"
                time.sleep(1 / 25)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @api.get("/state")
    def state():
        """Notes currently in view, their total and the recent conversation."""
        notes = list(sense.notes)
        return {"notes": notes, "total": sum(notes), "fps": round(sense.fps, 1), "conf": sense.conf,
                "log": list(sense.log)}

    @api.post("/settings")
    def settings(new: Settings):
        """Change the detection threshold while running: lower finds more notes, higher makes fewer mistakes."""
        sense.conf = min(max(new.conf, 0.05), 0.95)
        return {"conf": sense.conf}

    @api.post("/detect")
    async def detect(file: UploadFile):
        """Upload an image and get back every note found, with boxes and the total."""
        image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        result, notes = sense.detect(image)
        boxes = [{"note": note_value(result.names[int(c)]), "confidence": round(float(p), 3),
                  "box": [round(float(v)) for v in xyxy]}
                 for c, p, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)]
        return {"notes": notes, "total": sum(notes), "boxes": boxes, "speech": describe(notes)}

    @api.post("/ask")
    def ask(question: Question):
        """Ask a question as text instead of speaking, e.g. "How much money do I have?"."""
        return {"answer": sense.hear(question.text)}

    return api


def main():
    parser = argparse.ArgumentParser()
    trained = ROOT / "models" / "notes.pt"
    parser.add_argument("--source", default="0", help="camera index, stream URL or 'scrcpy'")
    parser.add_argument("--model", default=str(trained) if trained.exists() else "yolo26s.pt")
    parser.add_argument("--conf", type=float, default=0.4, help="best F1 on the test set was at 0.38")
    parser.add_argument("--imgsz", type=int, default=640, help="larger finds notes further away, at a lower FPS")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--voice", default="en-US-AndrewMultilingualNeural",
                        help="e.g. en-US-AvaMultilingualNeural for a female voice")
    parser.add_argument("--whisper", default="small", help="tiny, base, small or medium")
    parser.add_argument("--no-mic", action="store_true")
    parser.add_argument("--mute", action="store_true")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    sense = NoteDetection(args.model, args.conf, args.imgsz)
    if not args.mute:
        sense.speaker = Speaker(args.voice)
    if not args.no_mic:
        print("Loading speech recognition ...")
        Listener(sense.hear, sense.speaker, model=args.whisper)
    threading.Thread(target=sense.run, args=(Camera(args.source),), daemon=True).start()

    url = f"http://localhost:{args.port}"
    print(f"Dashboard: {url}")
    threading.Timer(2, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_api(sense), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
