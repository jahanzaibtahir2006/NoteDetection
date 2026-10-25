"""Step 1: build a labeled detection dataset from unlabeled photos.

    python prepare_data.py

1. Downloads the Pakistani Currency Notes dataset from Kaggle (3,611 photos, CC0).
   The photos are sorted into folders such as 500_front, but have no boxes.
2. Adds your own webcam frames from data/captured/<note>/ (see capture.py).
3. Auto-labels every new photo with a zero-shot detector prompted with "banknote".
   The class comes from the folder name. Boxes are cached in data/labels.json.
   - Kaggle close-ups on plain backgrounds hold one note each: YOLOE-26 (fast).
   - Captured webcam frames show a busy room and often several notes: Grounding DINO
     (slower, far fewer false boxes). Every note is boxed; boxes that wrap several
     notes, boxes that sit still across many frames (objects on the desk, not a
     note in a moving hand) and boxes too long and thin to be a note are dropped.
4. Drops photos without a reliable box, splits each group 80/10/10, writes
   data/notes/data.yaml and saves preview images for a visual check.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import urllib.request
import zipfile

import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLOE

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ARCHIVE = DATA / "pkr_kaggle.zip"
RAW = DATA / "raw"
CAPTURED = DATA / "captured"
CACHE = DATA / "labels.json"
OUT = DATA / "notes"
SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/muhammadammarabid/pakistanicurrencydataset"

# Kaggle close-ups: YOLOE with broad words finds folded or blurred notes too
YOLOE_WEIGHTS = "yoloe-26l-seg.pt"
YOLOE_PROMPTS = ["banknote", "paper money", "currency note", "money", "paper", "card"]
YOLOE_MIN_CONF = 0.08               # below this the box was wrong in manual review
# Captured frames: Grounding DINO
GDINO_WEIGHTS = "IDEA-Research/grounding-dino-base"
GDINO_PROMPT = "a banknote."
GDINO_MIN_CONF = 0.27               # group boxes around several notes score just below this

AREA = {"kaggle": (0.02, 0.9),      # close-up photos
        "captured": (0.002, 0.4)}   # webcam frames include notes held far away
SAME_BOX_IOU = 0.5                  # overlap above which two boxes are the same note
STILL_IOU, STILL_SHARE = 0.8, 0.08  # a box in the same spot in 8% of a session is not in a hand
MAX_ASPECT = 3.5                    # a note is about 2.2 times as long as it is wide; allow for tilt
SPLITS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def download():
    if not ARCHIVE.exists():
        DATA.mkdir(exist_ok=True)
        print("Downloading dataset from Kaggle (180 MB) ...")
        urllib.request.urlretrieve(SOURCE_URL, ARCHIVE)
    if not RAW.exists():
        print("Unpacking ...")
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(RAW)


def find_photos():
    """Return (path, note, source) for every photo, with the note value from its folder."""
    photos = [(p, int(p.parent.name.split("_")[0]), "kaggle") for p in sorted(RAW.rglob("*.jpg"))]
    if CAPTURED.exists():
        photos += [(p, int(p.parent.name), "captured") for p in sorted(CAPTURED.glob("*/*.jpg"))]
    return photos


def iou(a, b):
    """Overlap of two [cx, cy, w, h] boxes, 0 to 1."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def covers(outer, inner):
    """Share of the inner box that lies inside the outer box."""
    ox1, oy1, ox2, oy2 = outer[0] - outer[2] / 2, outer[1] - outer[3] / 2, outer[0] + outer[2] / 2, outer[1] + outer[3] / 2
    ix1, iy1, ix2, iy2 = inner[0] - inner[2] / 2, inner[1] - inner[3] / 2, inner[0] + inner[2] / 2, inner[1] + inner[3] / 2
    inter = max(0, min(ox2, ix2) - max(ox1, ix1)) * max(0, min(oy2, iy2) - max(oy1, iy1))
    return inter / (inner[2] * inner[3] + 1e-9)


class Labeler:
    def __init__(self):
        self.yoloe = self.gdino = None

    def boxes(self, photo, source):
        return self._kaggle(photo) if source == "kaggle" else self._captured(photo)

    def _kaggle(self, photo):
        """The single most confident note box in a close-up photo."""
        if not self.yoloe:
            self.yoloe = YOLOE(YOLOE_WEIGHTS)
            self.yoloe.set_classes(YOLOE_PROMPTS)
        result = self.yoloe.predict(str(photo), conf=0.03, imgsz=960, verbose=False)[0]
        low, high = AREA["kaggle"]
        found = [(float(c), [round(float(v), 6) for v in b]) for c, b in zip(result.boxes.conf, result.boxes.xywhn)
                 if c >= YOLOE_MIN_CONF and low < float(b[2] * b[3]) < high]
        return [max(found)[1]] if found else []

    def _captured(self, photo):
        """Every note in a webcam frame."""
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if not self.gdino:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.gdino = (AutoProcessor.from_pretrained(GDINO_WEIGHTS),
                          AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_WEIGHTS).to(device).eval())
        processor, model = self.gdino
        image = Image.open(photo).convert("RGB")
        W, H = image.size
        inputs = processor(images=image, text=GDINO_PROMPT, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=GDINO_MIN_CONF, text_threshold=GDINO_MIN_CONF,
            target_sizes=[(H, W)])[0]

        low, high = AREA["captured"]
        found = []
        for score, (x1, y1, x2, y2) in zip(result["scores"].tolist(), result["boxes"].tolist()):
            box = [round((x1 + x2) / 2 / W, 6), round((y1 + y2) / 2 / H, 6), round((x2 - x1) / W, 6), round((y2 - y1) / H, 6)]
            if low < box[2] * box[3] < high:
                found.append((score, box))
        kept = []
        for _, box in sorted(found, reverse=True):  # keep the strongest of overlapping boxes
            if all(iou(box, k) < SAME_BOX_IOU for k in kept):
                kept.append(box)
        # a box that wraps two or more other boxes is a group of notes, not one note
        return [b for b in kept if sum(covers(b, o) > 0.8 for o in kept if o is not b) < 2]


def drop_still_boxes(labeled):
    """Remove boxes that stay in one spot across a capture session: desk objects, not notes."""
    sessions = defaultdict(list)
    for item in labeled:
        if item[2] == "captured":
            sessions[item[1]].append(item[3])
    removed = 0
    for frames in sessions.values():
        for boxes in frames:
            for box in list(boxes):
                repeats = sum(any(iou(box, other) > STILL_IOU for other in others) for others in frames)
                if repeats > STILL_SHARE * len(frames):
                    boxes.remove(box)
                    removed += 1
    print(f"Dropped {removed} boxes that never moved")
    return [item for item in labeled if item[3]]


def drop_odd_shapes(labeled):
    """Remove captured boxes no handheld note makes: too long and thin, or running from the
    top edge of the frame to the bottom (the door frame)."""
    from PIL import Image

    removed = 0
    for photo, _, source, boxes in labeled:
        if source != "captured":
            continue
        W, H = Image.open(photo).size
        for box in list(boxes):
            long_side, short_side = sorted((box[2] * W, box[3] * H), reverse=True)
            full_height = box[1] - box[3] / 2 < 0.02 and box[1] + box[3] / 2 > 0.98
            if long_side > MAX_ASPECT * short_side or full_height:
                boxes.remove(box)
                removed += 1
    print(f"Dropped {removed} boxes with a shape no note has")
    return [item for item in labeled if item[3]]


def auto_label(photos):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = lambda p: p.relative_to(DATA).as_posix()
    todo = [(p, s) for p, _, s in photos if key(p) not in cache]
    if todo:
        print(f"Auto-labeling {len(todo)} new photos ({len(photos) - len(todo)} cached) ...")
        labeler = Labeler()
        for i, (photo, source) in enumerate(todo, 1):
            cache[key(photo)] = labeler.boxes(photo, source)
            if i % 250 == 0 or i == len(todo):
                print(f"  labeled {i}/{len(todo)}")
                CACHE.write_text(json.dumps(cache))
    labeled = [(p, n, s, [list(b) for b in cache[key(p)]]) for p, n, s in photos if cache[key(p)]]
    return drop_odd_shapes(drop_still_boxes(labeled))


def save_preview(samples, names, path):
    tiles = []
    for photo, cls, boxes in samples:
        image = cv2.imread(str(photo))
        H, W = image.shape[:2]
        for cx, cy, w, h in boxes:
            x1, y1, x2, y2 = int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)
            cv2.rectangle(image, (x1, y1), (x2, y2), (80, 230, 120), max(3, W // 150))
            cv2.putText(image, f"Rs {names[cls]}", (x1 + 10, max(y1 - 15, 50)), cv2.FONT_HERSHEY_SIMPLEX,
                        W / 450, (80, 230, 120), max(2, W // 300))
        tiles.append(cv2.resize(image, (320, 240)))
    while len(tiles) % 6:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def main():
    download()
    photos = find_photos()
    names = sorted({note for _, note, source in photos if source == "kaggle"})
    index = {note: i for i, note in enumerate(names)}
    print(f"{len(photos)} photos, notes: {names}")
    labeled = auto_label(photos)

    if OUT.exists():
        shutil.rmtree(OUT)
    groups = defaultdict(list)
    for photo, note, source, boxes in labeled:
        groups[(source, note)].append((photo, index[note], boxes))

    random.seed(0)
    counts = defaultdict(Counter)
    for (source, note), items in sorted(groups.items()):
        random.shuffle(items)
        n_train, n_valid = int(len(items) * SPLITS["train"]), int(len(items) * SPLITS["valid"])
        parts = {"train": items[:n_train], "valid": items[n_train:n_train + n_valid], "test": items[n_train + n_valid:]}
        for split, part in parts.items():
            (OUT / split / "images").mkdir(parents=True, exist_ok=True)
            (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
            for photo, cls, boxes in part:
                name = f"{source}_{note}_{photo.parent.name}_{photo.stem}"
                shutil.copy2(photo, OUT / split / "images" / f"{name}.jpg")
                (OUT / split / "labels" / f"{name}.txt").write_text(
                    "".join(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes))
            counts[split][(source, note)] = len(part)

    found = Counter((s, n) for _, n, s, _ in labeled)
    notes = Counter()
    for _, n, s, boxes in labeled:
        notes[(s, n)] += len(boxes)
    total = Counter((s, n) for _, n, s in photos)
    print(f"\n{'source':<10}{'note':>6}{'photos':>8}{'kept':>6}{'boxes':>7}{'train':>7}{'valid':>7}{'test':>6}")
    for key in sorted(total):
        print(f"{key[0]:<10}{key[1]:>6}{total[key]:>8}{found[key]:>6}{notes[key]:>7}"
              f"{counts['train'][key]:>7}{counts['valid'][key]:>7}{counts['test'][key]:>6}")
    print(f"{'all':<10}{'':>6}{len(photos):>8}{len(labeled):>6}{sum(notes.values()):>7}"
          f"{sum(counts['train'].values()):>7}{sum(counts['valid'].values()):>7}{sum(counts['test'].values()):>6}")

    config = {"path": str(OUT.resolve()), "train": "train/images", "val": "valid/images",
              "test": "test/images", "names": {i: str(n) for i, n in enumerate(names)}}
    (OUT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    for source in ("kaggle", "captured"):
        items = [item for (s, _), group in groups.items() if s == source for item in group]
        if items:
            path = OUT / f"preview_{source}.jpg"
            save_preview(random.sample(items, min(36, len(items))), names, path)
            print(f"Check the boxes: {path}")
    print(f"\nDataset ready: {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
