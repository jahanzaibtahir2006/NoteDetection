"""Step 2: fine-tune YOLO26 on Pakistani banknotes.

    python train.py               # 10 epochs (~1 hour on a GTX 1050 Ti)
    python train.py --epochs 40   # longer run for a little more accuracy
    python train.py --resume      # continue a run that was interrupted

Starts from models/notes.pt when it exists (so new captured photos refine the
current model), otherwise from the pretrained yolo26s. The best checkpoint is
copied to models/notes.pt.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    weights = ROOT / "models" / "notes.pt"
    parser.add_argument("--model", default=str(weights) if weights.exists() else "yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=512, help="notes fill much of the frame, so 512 is enough")
    parser.add_argument("--batch", type=int, default=8, help="8 at 512 px fits a 4 GB GPU; raise it on bigger cards")
    parser.add_argument("--workers", type=int, default=0, help="0 loads data in the main process; each extra worker costs ~500 MB of RAM on Windows")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run from its last epoch")
    args = parser.parse_args()

    data = ROOT / "data" / "notes" / "data.yaml"
    if not data.exists():
        raise SystemExit("Run `python prepare_data.py` first.")

    if args.resume:
        YOLO(ROOT / "runs" / "notes" / "weights" / "last.pt").train(resume=True)
    else:
        YOLO(args.model).train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=20,       # stop early once validation stops improving
            amp=False,         # FP16 is very slow on GTX 10-series GPUs
            workers=args.workers,
            project=str(ROOT / "runs"),
            name="notes",
            exist_ok=True,
            # Augmentation tuned for a handheld camera in a shop
            degrees=15,        # notes held at an angle
            fliplr=0.0,        # a real note never appears mirrored
            hsv_v=0.5,         # bright and dim lighting
            scale=0.9,         # shrink photos down to 10% so close-ups also teach far-away notes
            mosaic=1.0,        # several notes in one frame
            close_mosaic=max(1, args.epochs // 8),  # the default of 10 would switch mosaic off for a whole short run
        )

    weights.parent.mkdir(exist_ok=True)
    shutil.copy(ROOT / "runs" / "notes" / "weights" / "best.pt", weights)

    split = "test" if (ROOT / "data" / "notes" / "test").exists() else "val"
    metrics = YOLO(weights).val(data=str(data), split=split)
    print(f"\n{split} mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    print(f"Model saved to {weights}")


if __name__ == "__main__":
    main()
