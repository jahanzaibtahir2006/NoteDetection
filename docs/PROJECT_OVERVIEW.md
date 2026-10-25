# Note Detection: Project Overview

A detailed walkthrough of how Note Detection was built. All numbers below are measured results from this project.

---

## 1. What it is

**Note Detection** is a real-time AI assistant that helps visually impaired people handle cash.
A camera (laptop webcam, phone, or one clipped to a cap) watches the notes in the user's hand.
Note Detection recognises every Pakistani banknote in view, adds them up, and speaks the result:

- First note: *"Note detected. This is a 1000 rupee note."*
- Note added: *"A 50 rupee note has been added. You have 2050 rupees."*
- Note removed: *"A 1000 rupee note has been removed. You have 1050 rupees."*

It recognises 7 denominations: Rs 10, 20, 50, 100, 500, 1000 and 5000, and several notes at once.

**Hardware:** Windows 11 PC with an NVIDIA GTX 1050 Ti (4 GB). Everything was trained and runs locally on this PC.
**Live speed:** about 30 FPS.

---

## 2. Architecture

```
Camera (webcam / phone)
   │
   ▼
YOLO26s, fine-tuned             → finds and classifies each note in every frame
   │
   ▼
ByteTrack + voting               → follows each note across frames, keeps its value steady
   │
   ▼
Announcement logic               → works out what changed and builds the sentence
   │
   ▼
Neural text-to-speech            → speaks it in a natural human voice
   │
FastAPI server                   → live dashboard, video stream and REST API
```

---

## 3. Tech stack and models used

| Purpose | Model / tool | Notes |
|---|---|---|
| Note detection (main model) | **Ultralytics YOLO26s** | Latest YOLO generation; fine-tuned on Pakistani notes |
| Auto-labeling Kaggle photos | **YOLOE-26** (Ultralytics, zero-shot) | Finds a "banknote" from a text prompt, no training needed |
| Auto-labeling my own webcam footage | **Grounding DINO base** (IDEA Research, Hugging Face Transformers) | Strongest open-vocabulary detector tested; far fewer false boxes |
| Tracking notes across frames | **ByteTrack** (multi-object tracker) | Gives each note an ID from frame to frame |
| Voice output | **Microsoft neural TTS** via edge-tts, voice `en-US-AndrewMultilingualNeural` | Warm, conversational voice |
| Voice input (optional) | **faster-whisper** (OpenAI Whisper small, int8 on CPU) | Understands Urdu or English questions such as "How much money do I have?" |
| Backend / API | **FastAPI + Uvicorn** | Dashboard, MJPEG live video, REST endpoints, Swagger docs |
| Frameworks | Python 3.12, PyTorch 2.5 (CUDA 12.1), OpenCV | |

---

## 4. Step-by-step process

### Step 1: Dataset sourcing
- Public datasets on Roboflow were reviewed first. The largest one boxed **parts** of each note (the digits, the portrait) instead of the whole note. That would make one note count as several, so it was rejected.
- Chosen dataset: **"Pakistani Currency Notes Image dataset" by Muhammad Ammar Abid on Kaggle (CC0)**. It has 3,611 handheld phone photos of all 7 notes, front and back.
- The catch: the photos are only sorted into folders (`500_front`, `1000_back`, ...). They have **no bounding boxes**, so they cannot train a detector as they are.

### Step 2: Automatic labeling with zero-shot detection (no manual labeling)
- **Zero-shot / open-vocabulary object detection:** YOLOE-26 was asked to find "banknote", "paper money" or "currency note" in every photo, which gave the box.
- **Weak supervision:** the class (Rs 10 to 5000) comes from the folder name, the box from the AI.
- **Quality control:** boxes below 0.08 confidence, or smaller than 2% or larger than 90% of the photo, are dropped. The 24 lowest-confidence boxes were reviewed by eye to set these limits.
- Result: about **97.5% of photos auto-labeled**. A random preview grid was checked visually, and 24 out of 24 were correct.
- **Stratified split:** 80% train, 10% validation, 10% test, done per denomination so every note is equally represented.

### Step 3: Training round 1 (transfer learning)
- **Transfer learning / fine-tuning:** start from YOLO26s pretrained on the COCO dataset, and teach it Pakistani notes.
- Settings tuned for a 4 GB GPU: 512 px images, batch 8, FP32 (the GTX 10-series is slow at FP16), 10 epochs, about 5.5 minutes per epoch.
- **Data augmentation:** rotation ±15°, brightness changes (dim and bright shops), zoom in and out, no mirror flips (a real note is never mirrored), and **mosaic** (4 photos stitched into one, so the model sees several notes together).
- Result on the test set: **mAP50 0.967, mAP50-95 0.853**.

### Step 4: Live testing revealed a real-world gap
- On the live webcam, notes were found only when held very close. Notes further away were missed.
- Reason: the Kaggle photos are mostly close-ups.

### Step 5: Collecting my own data
- I wrote a capture tool (`capture.py`). It records from the webcam and saves a frame at random short intervals (0.25 to 0.7 s), for 90 seconds per note.
- I recorded the Rs 50 and Rs 1000 notes, moving them near and far, tilting and flipping them, often holding two notes at once.
- Result: **371 frames** in my own room and lighting.

### Step 6: Labeling the hard footage
- This footage is much harder: a busy room, posters, a desk, people walking by, and several notes per frame.
- Attempt 1, YOLOE: it put boxes on the shirt pocket and desk objects. **Rejected.**
- Attempt 2, zoom-in check with my own trained model: it did not remove the false boxes and lost real notes. **Rejected.**
- Attempt 3, **Grounding DINO** (prompt: "a banknote."): clean boxes around every note in the hands. **Adopted.**
- Extra automatic clean-up rules:
  - **Non-maximum suppression (NMS):** keep one box per note.
  - **Group-box removal:** drop boxes that wrap two or more notes together.
  - **Static-object filter (temporal consistency):** a note in a moving hand changes position. A box that stays in the same spot across many frames is a desk object and is removed.
  - **Shape filter:** drop shapes no note has, such as a door edge running the full frame height.
- Labels are cached, so only new photos are labeled on later runs.
- Result: **346 frames with 465 note boxes** added. Final dataset: **3,956 photos, 4,075 boxes**.

### Step 7: Training round 2 (fine-tuning on real-world data)
- Continued from the round-1 model with the new data, plus a stronger **scale augmentation** (photos shrunk to 10% to simulate distance).
- Stopped after 6 epochs, the best epoch.
- Results:

| Test | Round 1 model | Round 2 model |
|---|---|---|
| Full test set mAP50 | 0.915 | **0.969** |
| Full test set recall | 0.905 | **0.960** |
| My own webcam frames, notes found (recall) | **19%** | **76%** |
| My own webcam frames, mAP50 | 0.227 | **0.786** |

### Step 8: Confidence threshold calibration
- The **F1 score** (the balance of precision and recall) was measured at every threshold. The best was **0.38**, so the default is 0.40.
- A **live slider** on the dashboard changes the threshold while the app runs. Lower finds more notes; higher makes fewer mistakes.

### Step 9: Making it stable in real time
- Problem: the model decides every frame on its own, so a blurred Rs 1000 could read as Rs 50 for a frame or two.
- Fix: **ByteTrack multi-object tracking** plus **confidence-weighted majority voting** over each note's last 15 frames. A new note must appear in at least 3 frames before it counts. This is **temporal smoothing**.
- Result on my recordings: wrong labels went from **5 to 0**.
- The total is only spoken once it has held steady. A note that leaves the frame for less than 2 seconds is not announced again.

### Step 10: Voice and dashboard
- Short, factual announcements with no greetings or small talk, focused only on what changed and the total.
- Light-theme web dashboard: live annotated video, the total ("You have Rs 2,050"), a chip for each note, the threshold slider and FPS.
- **REST API** (FastAPI): `POST /detect` (send an image, get notes and a total as JSON), `GET /state`, `POST /settings`, `POST /ask`, and auto-generated **Swagger docs** at `/docs`.
- Camera sources: laptop webcam, phone over DroidCam or IP Webcam, or the phone camera mirrored with scrcpy.

---

## 5. Challenges and how they were solved

| Challenge | Solution |
|---|---|
| No labeled Pakistani note dataset that boxes whole notes | Auto-labeled an unlabeled Kaggle dataset with zero-shot detection |
| Notes far from the camera were missed | Recorded my own footage, plus scale augmentation |
| Auto-labels in a busy room were wrong (pocket, desk) | Switched to Grounding DINO, plus temporal and shape filters |
| 4 GB GPU ran out of memory | 512 px images, batch 8, FP32 |
| Windows ran out of RAM during training | Data loading in the main process (0 workers) |
| Label flicker (1000 read as 50) | ByteTrack plus 15-frame voting |
| Too many or too few detections | F1-calibrated threshold plus a live slider |

---

## 6. Project files

| File | Role |
|---|---|
| `prepare_data.py` | Download, auto-label, clean, split and preview the data |
| `capture.py` | Record my own training frames from the webcam |
| `train.py` | Fine-tune YOLO26 (can resume an interrupted run) |
| `app.py` | Live detection, tracking, voice and the FastAPI server |
| `assistant.py` | What Note Detection says |
| `voice.py` | Text-to-speech and speech recognition |
| `camera.py` | Webcam, stream URL or scrcpy input |
| `dashboard.html` | Live web dashboard |

---

**Dataset credit:** Pakistani Currency Notes Image dataset by Muhammad Ammar Abid, Kaggle, CC0.
