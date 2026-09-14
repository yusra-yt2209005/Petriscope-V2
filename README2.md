# PetriScope embryo counter — P2 head + tiled (SAHI) inference

Two changes, stacked, targeting your specific failure mode (detector saturates above ~200 embryos):

1. **`train_p2.py`** — trains a `-p2` variant (adds a stride-4 detection head so
   the network resolves 5-9px objects) on top of YOLO26n (NMS-free head +
   small-target-aware label assignment). This fixes the *architectural* limit
   on how small an object the network can represent at all.
2. **`infer_sahi.py`** — wraps the trained model in SAHI's sliced inference.
   This fixes the *density* limit: instead of one forward pass over the whole
   dish, it runs the model over overlapping tiles and merges results, so
   embryos packed closer than one grid cell apart no longer collide.

Neither alone is likely to get you cleanly to 1000; do both.

## Setup
```bash
pip install -r requirements.txt
```
If your installed `ultralytics` version predates YOLO26, use `--arch yolo11n-p2.yaml`
in the train script instead — you still get the P2 head, just without the
NMS-free head and STAL label assignment.

## 1. Data prep (the part scripts can't do for you)
- Re-annotate (or re-export existing annotations) at **native resolution**, not
  downscaled to 640 — the P2 head only helps if training data actually has the
  pixel detail to learn from.
- Make sure your training set includes images with the *high-density* regime
  (hundreds of embryos), not just sparse examples. If your current dataset
  tops out around 100-200 annotated embryos per image, the model has never
  seen the 1000-embryo distribution you need it to generalize to — that alone
  could be your whole ceiling, independent of architecture.
- If you train on tiles directly (recommended for consistency with inference),
  slice your labeled images into the same tile size you'll use at inference
  (`slice-size` in `infer_sahi.py`) so train/inference resolution match.

## 2. Train
```bash
python train_p2.py --data data.yaml --epochs 150 --imgsz 1280
```
Outputs `petriscope_runs/embryo_p2/weights/best.pt`.

## 3. Count with tiled inference
```bash
python infer_sahi.py --weights petriscope_runs/embryo_p2/weights/best.pt \
    --dir ./dish_images --out counts.csv --viz-dir ./viz
```

## Tuning if counts are still off
| Symptom | Try |
|---|---|
| Still undercounting at high density | decrease `--slice-size`, increase `--overlap` |
| Overcounting (duplicates at tile seams) | increase `--overlap`, lower `postprocess_match_threshold` in `infer_sahi.py` |
| Missing very faint/low-contrast embryos | lower `--conf` |
| Slow (many tiles × many dishes) | increase `--slice-size` back up, or batch tiles on GPU (`--device cuda:0`) |
| Counts plateau near a suspicious round number | check `max_det` wasn't silently truncating — it's set to 2000 in both scripts, raise further if needed |

## If this still isn't enough for 1000+ dense, touching embryos
Detection + NMS-style counting fundamentally struggles as objects start
touching/overlapping at scale, no matter how you tile it. The next step past
this kit is a density-map or point-regression counter (e.g. P2PNet-style),
which predicts a count/density field directly instead of discrete boxes and
doesn't degrade the same way under heavy overlap. Worth prototyping if you
hit another wall after this.
