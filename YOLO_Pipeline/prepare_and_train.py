"""
Rust Dataset Preparation + YOLOv8-seg Training Pipeline
=======================================================
Steps:
  1. Parse annotation overlays from pipeline_rust_annotated_*.png
     - Yellow/orange  (H 20-45)  → class 0: "rust"
     - Cyan/teal      (H 95-115) → class 1: "corrosion_zone" (secondary)
  2. Split 80/20 train/val
  3. Build YOLO dataset folder structure + data.yaml
  4. Train yolov8n-seg for 50 epochs at imgsz=320 (CPU-friendly)
  5. Copy best.pt → YOLO_Pipeline/weights/best.pt
"""

import os, sys, json, shutil, random, math

_extra = r"D:\pip_packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
RUST_DIR    = r"D:\Pipeline-Digital-Twin\Data\Rust_Images"
YOLO_DIR    = r"D:\Pipeline-Digital-Twin\YOLO_Pipeline"
DATASET_DIR = os.path.join(YOLO_DIR, "dataset")
WEIGHTS_DIR = os.path.join(YOLO_DIR, "weights")

CLASSES = {0: "rust", 1: "corrosion_zone"}

# HSV ranges for annotation overlays
OVERLAY_RANGES = [
    # (class_id, h_lo, h_hi, s_lo, v_lo)  — rust / amber / orange / yellow highlight
    (0, 10, 45,  100, 70),
    # cyan/teal secondary corrosion zone
    (1, 90, 125, 90,  70),
]

TRAIN_SPLIT = 0.80
IMGSZ       = 320
EPOCHS      = 60
BATCH       = 4          # small batch for low RAM
MIN_CONTOUR_AREA = 150   # px²  — ignore tiny noise regions


# ── 1. Extract polygon labels from annotation overlays ────────────────────────
def extract_labels(img_path):
    """Return list of (class_id, [(x_norm, y_norm), ...]) polygons."""
    img = cv2.imread(img_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    labels = []

    for (cls_id, h_lo, h_hi, s_lo, v_lo) in OVERLAY_RANGES:
        mask = cv2.inRange(hsv,
                           (h_lo, s_lo, v_lo),
                           (h_hi, 255,  255))
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            # Simplify polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx  = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
            pts_norm = [(float(p[0][0]) / w, float(p[0][1]) / h)
                        for p in approx]
            labels.append((cls_id, pts_norm))

    return labels


def label_to_yolo_line(cls_id, pts_norm):
    flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts_norm)
    return f"{cls_id} {flat}"


# ── 2. Build dataset folder structure ────────────────────────────────────────
def build_dataset(images_with_labels):
    for split in ("train", "val"):
        split_dir = os.path.join(DATASET_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(DATASET_DIR, split, sub), exist_ok=True)

    random.seed(42)
    random.shuffle(images_with_labels)
    n_train = max(1, math.floor(len(images_with_labels) * TRAIN_SPLIT))
    splits  = {
        "train": images_with_labels[:n_train],
        "val":   images_with_labels[n_train:],
    }

    stats = {}
    for split, items in splits.items():
        count = 0
        for img_path, labels in items:
            fname = os.path.basename(img_path)
            stem  = os.path.splitext(fname)[0]

            # Copy image
            dst_img = os.path.join(DATASET_DIR, split, "images", fname)
            shutil.copy2(img_path, dst_img)

            # Write label file
            dst_lbl = os.path.join(DATASET_DIR, split, "labels", stem + ".txt")
            with open(dst_lbl, "w") as f:
                for (cls_id, pts) in labels:
                    f.write(label_to_yolo_line(cls_id, pts) + "\n")
            count += 1
        stats[split] = count
        print(f"  {split}: {count} images")

    return stats


# ── 3. Write data.yaml ────────────────────────────────────────────────────────
def write_yaml():
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    train_path = os.path.join(DATASET_DIR, "train", "images").replace("\\", "/")
    val_path   = os.path.join(DATASET_DIR, "val",   "images").replace("\\", "/")
    nc = len(CLASSES)
    names_str  = "[" + ", ".join(f"'{v}'" for v in CLASSES.values()) + "]"
    content = (
        f"train: {train_path}\n"
        f"val:   {val_path}\n"
        f"nc:    {nc}\n"
        f"names: {names_str}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  data.yaml → {yaml_path}")
    return yaml_path


# ── 4. Train ──────────────────────────────────────────────────────────────────
def train(yaml_path):
    # Import here so env vars are already set
    import ultralytics.cfg
    from ultralytics import YOLO

    # Monkey-patch to swallow unknown keys (grayscale etc.)
    _orig = ultralytics.cfg.check_dict_alignment
    def _patched(base, custom, e=None, allowed_custom_keys=None):
        custom.pop("grayscale", None)
        return _orig(base, custom, e, allowed_custom_keys)
    ultralytics.cfg.check_dict_alignment = _patched

    print(f"\nInitialising YOLOv8n-seg model …")
    model = YOLO("yolov8n-seg.pt")

    runs_dir = os.path.join(YOLO_DIR, "runs")
    print(f"Starting training  epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}")
    model.train(
        data      = yaml_path,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        device    = "cpu",
        workers   = 0,          # single-threaded data loading
        # Augmentations tuned for IR/thermal-style rust imagery
        hsv_h     = 0.015,
        hsv_s     = 0.4,
        hsv_v     = 0.4,
        fliplr    = 0.5,
        flipud    = 0.1,
        degrees   = 10.0,
        scale     = 0.4,
        mosaic    = 0.5,
        project   = runs_dir,
        name      = "rust_seg_train",
        exist_ok  = True,
        verbose   = True,
    )

    # Copy best weights
    best_src = os.path.join(runs_dir, "rust_seg_train", "weights", "best.pt")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_dst = os.path.join(WEIGHTS_DIR, "best.pt")
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print(f"\n[OK] Best weights saved -> {best_dst}")
    else:
        print(f"\n[WARN] best.pt not found at {best_src}. Check runs/ dir.")

    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("  Rust Dataset Prep + YOLOv8-seg Training")
    print(f"{'='*60}\n")

    # 1. Extract labels
    print("Step 1 — Extracting annotation overlays …")
    rust_imgs = sorted([
        os.path.join(RUST_DIR, f)
        for f in os.listdir(RUST_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"  Found {len(rust_imgs)} annotated images.")

    images_with_labels = []
    skipped = 0
    for ip in rust_imgs:
        labels = extract_labels(ip)
        if labels:
            images_with_labels.append((ip, labels))
            print(f"  {os.path.basename(ip)}: {len(labels)} region(s)")
        else:
            print(f"  {os.path.basename(ip)}: [WARN] no overlay detected - skipping")
            skipped += 1

    print(f"\n  Usable images: {len(images_with_labels)}  |  Skipped: {skipped}")
    if not images_with_labels:
        print("[ERROR] No labels could be extracted. Aborting.")
        sys.exit(1)

    # 2+3. Dataset structure + yaml
    print("\nStep 2 — Building YOLO dataset structure …")
    build_dataset(images_with_labels)
    print("\nStep 3 — Writing data.yaml …")
    yaml_path = write_yaml()

    # 4. Train
    print("\nStep 4 — Training …")
    model = train(yaml_path)

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"  Weights → {os.path.join(WEIGHTS_DIR, 'best.pt')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
