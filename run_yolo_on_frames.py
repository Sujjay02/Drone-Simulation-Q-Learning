#!/usr/bin/env python3
"""
run_yolo_on_frames.py
---------------------
Runs Sid's YOLO disc-detection model on captured Gazebo frames and reports how
well it actually fires on imagery from your simulation.

Runs on the HOST machine (not the Apptainer container), using the project venv
that already has torch installed.

Usage:
    cd ~/Drone-Simulation-Q-Learning
    source venv/bin/activate
    pip install ultralytics  # one-time, if not yet installed

    python3 run_yolo_on_frames.py \\
        --model ~/path/to/best.pt \\
        --input ~/yolo_test_frames \\
        --conf 0.25

    # Sweep multiple confidence thresholds to find a working one
    python3 run_yolo_on_frames.py --model best.pt --input ~/yolo_test_frames --sweep
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def run_at_conf(model, images, conf, imgsz):
    """Run inference at a given confidence threshold and return per-image results."""
    results = []
    for img_path in images:
        r = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        n = len(r.boxes)
        confs = r.boxes.conf.tolist() if n > 0 else []
        results.append({
            "image": img_path.name,
            "path": str(img_path),
            "detections": n,
            "confidences": [round(c, 3) for c in confs],
            "result_obj": r,
        })
    return results


def print_summary(results, conf):
    total = sum(r["detections"] for r in results)
    all_confs = [c for r in results for c in r["confidences"]]
    with_dets = sum(1 for r in results if r["detections"] > 0)
    n = len(results)

    print(f"\n--- conf={conf} ---")
    print(f"  Total detections:      {total}")
    print(f"  Avg per image:         {total / n:.2f}")
    print(f"  Images with ≥1 det:    {with_dets}/{n} ({100 * with_dets / n:.0f}%)")
    if all_confs:
        print(f"  Confidence range:      {min(all_confs):.2f} - {max(all_confs):.2f}")
        print(f"  Mean confidence:       {sum(all_confs) / len(all_confs):.2f}")


def save_annotated(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        annotated = r["result_obj"].plot()
        cv2.imwrite(str(output_dir / r["image"]), annotated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--input", required=True, help="Directory of captured PNGs")
    parser.add_argument(
        "--output", default=None,
        help="Where to save annotated images (default: <input>/annotated)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Try multiple confidence thresholds (0.05, 0.10, 0.25, 0.40)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else input_dir / "annotated"

    images = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ])
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Classes: {model.names}")
    print(f"Device:  {model.device}")
    print(f"Found {len(images)} images.")

    if args.sweep:
        # Run at multiple thresholds to find a sensible operating point
        for conf in (0.05, 0.10, 0.25, 0.40):
            results = run_at_conf(model, images, conf, args.imgsz)
            print_summary(results, conf)
        # Save annotated using the most permissive run so we can see borderline detections
        results = run_at_conf(model, images, 0.05, args.imgsz)
        save_annotated(results, output_dir)
        print(f"\nAnnotated images (conf=0.05): {output_dir}")
        return

    results = run_at_conf(model, images, args.conf, args.imgsz)
    for r in results:
        print(f"  {r['image']}: {r['detections']} detections | "
              f"conf={[round(c, 2) for c in r['confidences']]}")
    print_summary(results, args.conf)

    save_annotated(results, output_dir)

    # Strip the non-serializable result objects before writing JSON
    log = [{k: v for k, v in r.items() if k != "result_obj"} for r in results]
    json_path = output_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nAnnotated images: {output_dir}")
    print(f"Detection log:    {json_path}")

    # Diagnostic read
    total = sum(r["detections"] for r in results)
    with_dets = sum(1 for r in results if r["detections"] > 0)
    print("\n--- Read ---")
    if total == 0:
        print("  Zero detections at this conf. Try:")
        print("    python3 run_yolo_on_frames.py --model best.pt --input <dir> --sweep")
        print("  If sweep also returns ~0, model likely doesn't generalize to Gazebo")
        print("  imagery. Options: retrain on captured frames, or ask Sid for the")
        print("  training dataset to see what domain it was trained on.")
    elif with_dets / len(results) < 0.5:
        print("  Inconsistent. Many frames missed. Lower --conf or capture frames")
        print("  closer to discs, then re-test.")
    else:
        print("  Model fires on Gazebo imagery. Ready to integrate into a perception")
        print("  node that fuses detections + drone pose into a world-frame disc map.")


if __name__ == "__main__":
    main()
