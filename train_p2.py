"""
Train a small-object-aware detector for PetriScope embryo counting.

Why this config instead of stock `best.pt`:
  - `*-p2.yaml` adds a stride-4 (P2) detection head, so the network reasons about
    features at ~4x4px granularity instead of bottoming out at stride-8 (P3).
    That's the difference between "sees a 6px embryo" and "never had the resolution to."
  - YOLO26 is NMS-free at inference (end-to-end head) and uses STAL
    (Small-Target-Aware Label Assignment) during training. Both matter here:
    stock NMS is exactly what silently deletes touching/overlapping embryos
    once they're packed close together.
  - imgsz is raised well above the 640 default. At 640, a 1280px-wide dish photo
    already halves every embryo's pixel footprint before the network sees it.
  - max_det is raised from Ultralytics' default (300) to something that won't
    truncate you at your target density (1000 embryos/dish + headroom).

Usage:
    python train_p2.py --data /path/to/data.yaml --epochs 150
"""

import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml (Ultralytics format)")
    ap.add_argument(
        "--arch",
        default="yolo26n-p2.yaml",
        choices=["yolo26n-p2.yaml", "yolo11n-p2.yaml"],
        help="yolo26n-p2 is preferred (NMS-free + STAL small-object label assignment). "
        "Fall back to yolo11n-p2 only if your Ultralytics version predates YOLO26.",
    )
    ap.add_argument("--imgsz", type=int, default=1280, help="train at native/tile resolution, not 640")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8, help="drop this if you OOM at imgsz=1280")
    ap.add_argument("--project", default="petriscope_runs")
    ap.add_argument("--name", default="embryo_p2")
    args = ap.parse_args()

    model = YOLO(args.arch)  # random init from the architecture YAML, not a pretrained .pt
    # If you have a same-arch pretrained checkpoint to warm-start from, load it here instead:
    # model = YOLO("yolo26n-p2.pt")  # only if such weights exist for your arch/version

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        max_det=2000,        # headroom above the 1000-embryo BRC requirement
        # --- augmentation tuned for tiny, densely packed, roughly circular objects ---
        mosaic=1.0,           # keep: helps the model see many dense-crop compositions
        close_mosaic=15,      # disable mosaic for the last N epochs to sharpen localization
        scale=0.2,            # limit random scale-down aug; embryos are already tiny, don't shrink further
        degrees=15.0,         # embryos have no canonical orientation, rotation aug is safe/helpful
        fliplr=0.5,
        flipud=0.5,
        copy_paste=0.3,       # synthesize extra density/overlap during training, matches your >200 regime
        hsv_h=0.0,            # keep color aug minimal if embryos are ~monochrome under your imaging setup
    )

    metrics = model.val(imgsz=args.imgsz, max_det=2000)
    print(metrics)


if __name__ == "__main__":
    main()
