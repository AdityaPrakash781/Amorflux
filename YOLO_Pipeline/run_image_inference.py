"""
Batch image inference for pipeline section images using YOLOv8-seg.

Processes all PNG/JPG images in Data/Live_Pipeline_Feed, applies CLAHE
preprocessing (same as the video pipeline), runs YOLOv8 segmentation, and
saves:
  - Annotated output images  → YOLO_Pipeline/results/images/
  - Per-image JSON results   → YOLO_Pipeline/results/detections.json
  - Summary report           → YOLO_Pipeline/results/summary.json
"""

import os
import sys
import json
import glob

# ── Inject D:\pip_packages so ultralytics & cv2 are found ───────────────────
_extra = r"D:\pip_packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

# pyrefly: ignore [missing-import]
from ultralytics import YOLO


# ── Colour-coding by severity (confidence) ──────────────────────────────────
def severity_colour(conf: float) -> tuple[int, int, int]:
    """BGR colour for overlay, scaled by confidence."""
    if conf >= 0.70:
        return (0, 0, 220)   # Red   – high severity
    elif conf >= 0.45:
        return (0, 165, 255) # Orange – moderate severity
    else:
        return (0, 215, 255) # Yellow – low / possible detection


def severity_label(conf: float) -> str:
    if conf >= 0.70:
        return "HIGH"
    elif conf >= 0.45:
        return "MODERATE"
    return "LOW"


# ── CLAHE low-light pre-processing (same as video pipeline) ─────────────────
def apply_clahe_lab(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB to enhance low-light defects."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ── Main inference loop ──────────────────────────────────────────────────────
def run_image_inference(image_dir: str, model_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Pipeline Section Image Inference")
    print(f"  Model  : {model_path}")
    print(f"  Images : {image_dir}")
    print(f"{'='*60}\n")

    # ── Load model ───────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"[WARN] Fine-tuned weights not found at:\n       {model_path}")
        fallback = "yolov8n-seg.pt"
        print(f"[INFO] Falling back to pretrained base model: {fallback}")
        model_path = fallback
    
    print(f"Loading YOLO model: {model_path} ...")
    model = YOLO(model_path)
    print("Model loaded.\n")

    # ── Discover images ──────────────────────────────────────────────────────
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pat)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[ERROR] No images found in: {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process.\n")

    # ── Output directory ─────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "results", "images")
    os.makedirs(out_dir, exist_ok=True)

    # ── Per-image processing ─────────────────────────────────────────────────
    all_results = []
    total_detections = 0

    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        print(f"[{idx:02d}/{len(image_paths)}] Processing: {filename}")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"       [SKIP] Could not read image.\n")
            continue

        # 1. CLAHE preprocessing
        processed = apply_clahe_lab(frame)

        # 2. YOLO inference
        yolo_results = model(processed, verbose=False)[0]

        # 3. Parse detections
        image_detections = []
        annotated = frame.copy()

        if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            boxes      = yolo_results.boxes
            confs      = boxes.conf.cpu().numpy()
            cls_ids    = boxes.cls.cpu().numpy().astype(int)
            xyxy       = boxes.xyxy.cpu().numpy()
            has_masks  = yolo_results.masks is not None
            masks_xy   = yolo_results.masks.xy if has_masks else []

            for i, (conf, cls_id, box) in enumerate(zip(confs, cls_ids, xyxy)):
                conf       = float(conf)
                class_name = model.names.get(int(cls_id), f"class_{cls_id}")
                colour     = severity_colour(conf)
                sev_lbl    = severity_label(conf)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                # Draw segmentation mask overlay if available
                mask_coords = []
                if has_masks and i < len(masks_xy):
                    pts = masks_xy[i]
                    mask_coords = [[int(p[0]), int(p[1])] for p in pts]
                    if len(pts) > 2:
                        poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                        overlay = annotated.copy()
                        cv2.fillPoly(overlay, [poly], colour)
                        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
                        cv2.polylines(annotated, [poly], True, colour, 2)

                # Label
                label = f"{class_name} {conf:.2f} [{sev_lbl}]"
                label_y = max(y1 - 8, 15)
                cv2.putText(annotated, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

                det = {
                    "detection_id"  : i + 1,
                    "class_id"      : int(cls_id),
                    "class_name"    : class_name,
                    "confidence"    : round(conf, 4),
                    "severity"      : sev_lbl,
                    "bbox_xyxy"     : [x1, y1, x2, y2],
                    "mask_coords"   : mask_coords,
                }
                image_detections.append(det)

                print(f"       └─ {class_name:20s} conf={conf:.3f}  severity={sev_lbl}")

        if not image_detections:
            print(f"       └─ No defects detected.")

        # 4. Save annotated image
        out_name = f"annotated_{filename}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, annotated)

        total_detections += len(image_detections)
        all_results.append({
            "image_file"       : filename,
            "detections_count" : len(image_detections),
            "defect_detected"  : len(image_detections) > 0,
            "detections"       : image_detections,
            "annotated_output" : out_path,
        })
        print()

    # ── Save full detections JSON ─────────────────────────────────────────────
    results_dir = os.path.join(script_dir, "results")
    detections_json_path = os.path.join(results_dir, "detections.json")
    with open(detections_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # ── Build & save summary ─────────────────────────────────────────────────
    images_with_defects = [r for r in all_results if r["defect_detected"]]
    
    # Find worst image (highest single detection confidence)
    worst_image = None
    worst_conf  = 0.0
    for r in all_results:
        for det in r["detections"]:
            if det["confidence"] > worst_conf:
                worst_conf  = det["confidence"]
                worst_image = r["image_file"]

    summary = {
        "total_images_processed"  : len(all_results),
        "total_detections"        : total_detections,
        "images_with_detections"  : len(images_with_defects),
        "detection_rate_pct"      : round(100 * len(images_with_defects) / max(len(all_results), 1), 1),
        "highest_confidence_image": worst_image,
        "highest_confidence_score": round(worst_conf, 4),
        "model_used"              : model_path,
        "annotated_images_dir"    : out_dir,
    }

    summary_json_path = os.path.join(results_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Images processed   : {summary['total_images_processed']}")
    print(f"  Total detections   : {summary['total_detections']}")
    print(f"  Images w/ defects  : {summary['images_with_detections']}")
    print(f"  Detection rate     : {summary['detection_rate_pct']}%")
    if worst_image:
        print(f"  Highest confidence : {worst_conf:.3f}  ({worst_image})")
    print(f"\n  Annotated images → {out_dir}")
    print(f"  detections.json  → {detections_json_path}")
    print(f"  summary.json     → {summary_json_path}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_imgs = os.path.join(base_dir, "Data", "Live_Pipeline_Feed")
    default_wts  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt")

    img_dir    = sys.argv[1] if len(sys.argv) > 1 else default_imgs
    model_path = sys.argv[2] if len(sys.argv) > 2 else default_wts

    run_image_inference(img_dir, model_path)

