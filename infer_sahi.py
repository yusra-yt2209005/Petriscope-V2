"""
Tiled inference for dense embryo counting.

Why tiling: even with a P2 head, a single 1280px forward pass on a full petri-dish
image still assigns each embryo only a handful of source pixels. SAHI slices the
full-resolution image into overlapping windows, runs the detector on each window
at effectively "zoomed in" resolution, then stitches + de-duplicates detections
across the overlaps. This is what lets a nano model count objects that are only
a few pixels wide across a large field of view.

Tune these two knobs first if you're still under/over-counting:
  --slice-size   smaller = more zoom-in on each embryo, but more tiles/slower.
                 Start near 2-3x your embryo diameter's "effective receptive field" -
                 640 is a reasonable default for 5-9px embryos on a several-MP image.
  --overlap      needs to be >= one embryo diameter (in slice-relative terms) so no
                 embryo is cut in half at every tile boundary. 0.2 (20%) is a safe start.

Usage:
    python infer_sahi.py --weights runs/embryo_p2/weights/best.pt --image dish001.png
    python infer_sahi.py --weights best.pt --dir ./dish_images --out counts.csv
"""

import argparse
import glob
import os

import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def build_model(weights_path, conf, device):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights_path,
        confidence_threshold=conf,
        device=device,  # "cpu" or "cuda:0"
    )


def count_one(model, image_path, slice_size, overlap, out_dir=None):
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        # postprocess merges detections that fall in tile-overlap regions;
        # NMS here is across the WHOLE image after stitching, not per-tile,
        # so touching-tile duplicates get merged correctly instead of double-counted.
        postprocess_type="NMS",
        postprocess_match_metric="IOS",   # intersection-over-smaller: better than IoU for tiny/round objects
        postprocess_match_threshold=0.3,
        verbose=0,
    )
    count = len(result.object_prediction_list)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        result.export_visuals(export_dir=out_dir, file_name=stem, hide_labels=True, hide_conf=True)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to trained best.pt")
    ap.add_argument("--image", help="single image path")
    ap.add_argument("--dir", help="directory of images (alternative to --image)")
    ap.add_argument("--out", default=None, help="CSV path to write per-image counts")
    ap.add_argument("--viz-dir", default=None, help="optional dir to save annotated images")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--slice-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = build_model(args.weights, args.conf, args.device)

    if args.image:
        paths = [args.image]
    elif args.dir:
        paths = sorted(glob.glob(os.path.join(args.dir, "*")))
    else:
        raise SystemExit("pass --image or --dir")

    rows = []
    for p in paths:
        c = count_one(model, p, args.slice_size, args.overlap, args.viz_dir)
        print(f"{p}: {c} embryos")
        rows.append({"image": p, "embryo_count": c})

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
