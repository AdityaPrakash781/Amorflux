"""
Lightweight batch image inference for pipeline sections using YOLOv8-seg.
Runs CPU-only with OpenBLAS/OMP thread limits to avoid OOM on low-RAM systems.
Processes images one at a time to minimise peak memory.
"""

import os, sys, json, glob

# ── D:\pip_packages path injection ──────────────────────────────────────────
_extra = r"D:\pip_packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

# ── Limit OpenBLAS / OMP threads BEFORE importing torch/numpy ───────────────
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np
from ultralytics import YOLO


# ── Severity helpers ─────────────────────────────────────────────────────────
def severity_colour(conf):
    if conf >= 0.70:  return (0,   0, 220)   # Red
    if conf >= 0.45:  return (0, 165, 255)   # Orange
    return                   (0, 215, 255)   # Yellow

def severity_label(conf):
    if conf >= 0.70: return "HIGH"
    if conf >= 0.45: return "MODERATE"
    return "LOW"


# ── CLAHE (same pre-processing as video pipeline) ────────────────────────────
def apply_clahe_lab(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)


# ── Main ─────────────────────────────────────────────────────────────────────
def run(image_dir, model_path):
    print(f"\n{'='*60}")
    print("  Pipeline Section Image Inference  (Lite / Low-RAM)")
    print(f"  Images : {image_dir}")
    print(f"{'='*60}\n")

    # Fallback model
    if not os.path.exists(model_path):
        print(f"[WARN] No fine-tuned weights at {model_path}")
        model_path = "yolov8n-seg.pt"
        print(f"[INFO] Using pretrained base model: {model_path}\n")

    print(f"Loading model: {model_path}  (CPU-only) ...")
    model = YOLO(model_path)
    model.to("cpu")
    print("Model ready.\n")

    # Discover images
    patterns = ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"]
    paths = sorted({p for pat in patterns for p in glob.glob(os.path.join(image_dir, pat))})
    if not paths:
        print(f"[ERROR] No images found in {image_dir}"); sys.exit(1)
    print(f"Found {len(paths)} image(s).\n")

    # Output dirs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, "results", "images")
    os.makedirs(out_dir, exist_ok=True)

    all_results   = []
    total_dets    = 0
    worst_img     = None
    worst_conf    = 0.0

    for idx, img_path in enumerate(paths, 1):
        fname = os.path.basename(img_path)
        print(f"[{idx:02d}/{len(paths)}] {fname}")

        frame = cv2.imread(img_path)
        if frame is None:
            print("       SKIP – unreadable\n"); continue

        processed  = apply_clahe_lab(frame)
        yolo_res   = model(processed, verbose=False, device="cpu", imgsz=320, conf=0.15)[0]
        annotated  = frame.copy()
        detections = []

        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            confs    = yolo_res.boxes.conf.cpu().numpy()
            cls_ids  = yolo_res.boxes.cls.cpu().numpy().astype(int)
            xyxy     = yolo_res.boxes.xyxy.cpu().numpy()
            has_mask = yolo_res.masks is not None
            masks_xy = yolo_res.masks.xy if has_mask else []

            for i, (conf, cls_id, box) in enumerate(zip(confs, cls_ids, xyxy)):
                conf       = float(conf)
                class_name = model.names.get(int(cls_id), f"class_{cls_id}")
                colour     = severity_colour(conf)
                sev        = severity_label(conf)

                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(annotated, (x1,y1), (x2,y2), colour, 2)

                mask_coords = []
                if has_mask and i < len(masks_xy):
                    pts = masks_xy[i]
                    mask_coords = [[int(p[0]), int(p[1])] for p in pts]
                    if len(pts) > 2:
                        poly = np.array(pts, np.int32).reshape((-1,1,2))
                        ov   = annotated.copy()
                        cv2.fillPoly(ov, [poly], colour)
                        cv2.addWeighted(ov, 0.35, annotated, 0.65, 0, annotated)
                        cv2.polylines(annotated, [poly], True, colour, 2)

                label = f"{class_name} {conf:.2f} [{sev}]"
                cv2.putText(annotated, label, (x1, max(y1-8,15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

                detections.append({
                    "detection_id" : i+1,
                    "class_id"     : int(cls_id),
                    "class_name"   : class_name,
                    "confidence"   : round(conf, 4),
                    "severity"     : sev,
                    "bbox_xyxy"    : [x1,y1,x2,y2],
                    "mask_coords"  : mask_coords,
                })
                print(f"       └─ {class_name:20s}  conf={conf:.3f}  [{sev}]")
                if conf > worst_conf:
                    worst_conf = conf; worst_img = fname

        if not detections:
            print("       └─ No defects detected.")

        out_path = os.path.join(out_dir, f"annotated_{fname}")
        cv2.imwrite(out_path, annotated)

        total_dets += len(detections)
        all_results.append({
            "image_file"       : fname,
            "detections_count" : len(detections),
            "defect_detected"  : bool(detections),
            "detections"       : detections,
            "annotated_output" : out_path,
        })
        print()

    # Save JSON outputs
    res_dir = os.path.join(script_dir, "results")
    det_path = os.path.join(res_dir, "detections.json")
    with open(det_path, "w") as f:
        json.dump(all_results, f, indent=2)

    images_with_dets = sum(1 for r in all_results if r["defect_detected"])
    summary = {
        "total_images_processed"   : len(all_results),
        "total_detections"         : total_dets,
        "images_with_detections"   : images_with_dets,
        "detection_rate_pct"       : round(100*images_with_dets/max(len(all_results),1), 1),
        "highest_confidence_image" : worst_img,
        "highest_confidence_score" : round(worst_conf, 4),
        "model_used"               : model_path,
        "annotated_images_dir"     : out_dir,
    }
    sum_path = os.path.join(res_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Images processed   : {summary['total_images_processed']}")
    print(f"  Total detections   : {summary['total_detections']}")
    print(f"  Images w/ defects  : {summary['images_with_detections']}")
    print(f"  Detection rate     : {summary['detection_rate_pct']}%")
    if worst_img:
        print(f"  Highest confidence : {worst_conf:.3f}  ({worst_img})")
    print(f"\n  Annotated images → {out_dir}")
    print(f"  detections.json  → {det_path}")
    print(f"  summary.json     → {sum_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    base       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir    = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "Data", "Live_Pipeline_Feed")
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt")
    run(img_dir, model_path)

