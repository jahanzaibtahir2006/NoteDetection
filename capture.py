"""Record your own training photos from the live camera.

    python capture.py 50 1000              # 90 seconds per note
    python capture.py 500 --seconds 60
    python capture.py 50 --source 1        # another camera

For each note a window opens. Press SPACE to start, then move the note around:
near and far, tilted, front and back, in and out of your hand. Frames are saved
at short random intervals to data/captured/<note>/. ESC stops early.
Then run `python prepare_data.py` to label them and `python train.py` to retrain.
"""
import argparse
from pathlib import Path
import random
import time

import cv2

from camera import Camera

OUT = Path(__file__).parent / "data" / "captured"
GAP = (0.25, 0.7)  # seconds between saved frames


def overlay(frame, lines, color=(80, 230, 120)):
    view = cv2.flip(frame, 1)  # mirror the preview so moving the note feels natural
    for i, text in enumerate(lines):
        cv2.putText(view, text, (20, 45 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(view, text, (20, 45 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    cv2.imshow("Note Detection capture", view)
    return cv2.waitKey(1) & 0xFF


def record(camera, note, seconds):
    folder = OUT / str(note)
    folder.mkdir(parents=True, exist_ok=True)
    while (key := overlay(camera.frame, [f"Next: Rs {note}", "Press SPACE to start, ESC to quit"], (0, 200, 255))) != 32:
        if key == 27:
            raise SystemExit
    saved, end, next_shot = 0, time.time() + seconds, time.time()
    while (left := end - time.time()) > 0:
        frame = camera.frame
        if time.time() >= next_shot:
            cv2.imwrite(str(folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{saved:04d}.jpg"), frame)
            saved += 1
            next_shot = time.time() + random.uniform(*GAP)
        key = overlay(frame, [f"Rs {note}   {int(left)}s left   {saved} saved",
                              "Move it near and far, tilt it, flip it"])
        if key == 27:
            break
    print(f"Rs {note}: saved {saved} frames to {folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notes", nargs="+", type=int, help="note values to record, e.g. 50 1000")
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--source", default="0")
    args = parser.parse_args()

    camera = Camera(args.source).wait()
    for note in args.notes:
        record(camera, note, args.seconds)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
