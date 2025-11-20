#!/usr/bin/env python3
"""
Evaluate Siamese detector on an entire samples/ dataset and export detections to JSON.

Data layout (per class/instance):

train/samples/
  Backpack_0/
    object_images/
      img_1.jpg
      img_2.jpg
      ...
    drone_video.mp4

Output JSON format:

[
  {
    "video_id": "Backpack_0",
    "detections": [
      {
        "bboxes": [
          {"frame": 370, "x1": 422, "y1": 310, "x2": 470, "y2": 355},
          {"frame": 371, "x1": 424, "y1": 312, "x2": 468, "y2": 354}
        ]
      }
    ]
  },
  {
    "video_id": "Jacket_0",
    "detections": []
  }
]

Optionally also writes visualization videos (with different thresholds) if --save_video is set.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2
import torch
import numpy as np
from tqdm import tqdm

from SM2 import build_siamese_model, make_support_proto, denormalize_for_model
from Data2 import IMAGENET_MEAN, IMAGENET_STD


# =========================================================
# SUPPORT IMAGE LOADING
# =========================================================

def load_support_images_as_tensor(paths: List[Path], device: torch.device) -> torch.Tensor:
    """
    Load support images and resize to 256x256 + normalize exactly like training.

    Returns:
        Tensor of shape (K, C, H, W) on given device.
    """
    SUPPORT_SIZE = 256
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)

    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Cannot read support: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SUPPORT_SIZE, SUPPORT_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype("float32") / 255.0
        img = (img - mean) / std  # normalize
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        out.append(t)

    return torch.stack(out, dim=0).to(device)  # (K, C, H, W)


# =========================================================
# QUERY FRAME PREPROCESSING
# =========================================================

def preprocess_frame(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert BGR → normalized tensor → then denormalized back to [0,1] to match model input.

    This preserves the same pipeline as training where they:
      - normalize
      - then use denormalize_for_model before passing to Faster R-CNN.
    """
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    norm = (rgb - mean) / std

    t = torch.from_numpy(norm).permute(2, 0, 1).float().to(device)
    t_0_1 = denormalize_for_model(t, device)  # back to [0,1], same as training
    return t_0_1  # (3, H, W)


# =========================================================
# DRAWING (FOR OPTIONAL VIDEO OUTPUT)
# =========================================================

def draw_threshold_boxes(frame: np.ndarray,
                         boxes: np.ndarray,
                         scores: np.ndarray,
                         threshold: float,
                         color,
                         label_prefix: str = "") -> int:
    """
    Draw boxes on frame with given threshold and color.
    Returns count of drawn boxes.
    """
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]

    count = 0
    for box, s in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix}{s:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        count += 1

    return count


# =========================================================
# EVAL ONE VIDEO
# =========================================================

def eval_single_video(
    model,
    support_paths: List[Path],
    video_path: Path,
    device: torch.device,
    score_threshold: float,
    resize_short_side: int | None = None,
    save_video: bool = False,
    video_out_path: Path | None = None,
    draw_thresholds: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate a single video with given support images.

    Returns:
        List of bbox dicts:
          [{"frame": int, "x1": int, "y1": int, "x2": int, "y2": int}, ...]
    """

    # Default drawing thresholds if saving video
    if draw_thresholds is None:
        draw_thresholds = [0.2, 0.4, 0.6]

    # 1. Compute prototype from supports
    support_tensors = load_support_images_as_tensor(support_paths, device)
    with torch.no_grad():
        proto = make_support_proto(model, support_tensors, device)
        model.roi_heads.box_predictor.set_prototype(proto)

    # 2. Video IO
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_short_side:
        short = min(width, height)
        scale = resize_short_side / float(short)
        width = int(width * scale)
        height = int(height * scale)

    writer = None
    if save_video:
        assert video_out_path is not None
        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    COLORS = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
    ]

    # 3. Iterate frames and collect detections
    all_bboxes: List[Dict[str, Any]] = []
    frame_idx = 0

    with torch.no_grad():
        pbar = tqdm(desc=f"Video {video_path.name}", unit="frame")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if resize_short_side:
                frame = cv2.resize(frame, (width, height))

            # Model input
            input_tensor = preprocess_frame(frame, device)  # (3,H,W)
            preds = model([input_tensor])[0]

            # Convert to CPU numpy
            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            # Keep only label==1
            keep_lbl = labels == 1
            boxes_fg = boxes[keep_lbl]
            scores_fg = scores[keep_lbl]

            # For JSON: keep boxes with score >= score_threshold
            keep_thr = scores_fg >= score_threshold
            boxes_json = boxes_fg[keep_thr]

            for b in boxes_json:
                x1, y1, x2, y2 = b.astype(int).tolist()
                all_bboxes.append(
                    {
                        "frame": frame_idx,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                )

            # Optional video visualization
            if writer is not None:
                frame_draw = frame.copy()
                for i, thr in enumerate(draw_thresholds):
                    c = COLORS[i % len(COLORS)]
                    count = draw_threshold_boxes(
                        frame_draw,
                        boxes_fg,
                        scores_fg,
                        thr,
                        c,
                        label_prefix=f"{thr:.2f}:",
                    )
                    cv2.putText(
                        frame_draw,
                        f"thr {thr:.2f}: {count}",
                        (10, 25 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        c,
                        1,
                    )

                cv2.putText(
                    frame_draw,
                    f"frame {frame_idx}",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                writer.write(frame_draw)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    if writer is not None:
        writer.release()

    return all_bboxes


# =========================================================
# MAIN EVAL OVER DATASET
# =========================================================

def run_eval_over_samples(
    checkpoint: Path,
    samples_root: Path,
    output_json: Path,
    device_name: str = "cuda",
    score_threshold: float = 0.5,
    save_video: bool = False,
    video_out_dir: Path | None = None,
    resize_short_side: int | None = None,
):
    """
    Iterate over samples/ folders, run detection, and export JSON.
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    print("Building Siamese model...")
    model = build_siamese_model(agnostic_weights=None)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    print("Loading checkpoint weights...")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Improve recall: let us handle threshold in our code
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh = 0.7
    model.roi_heads.detections_per_img = 300

    # Iterate over sample folders
    samples_root = samples_root.resolve()
    if save_video:
        assert video_out_dir is not None
        video_out_dir = video_out_dir.resolve()
        video_out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    sample_dirs = sorted([d for d in samples_root.iterdir() if d.is_dir()])
    print(f"Found {len(sample_dirs)} sample folders under {samples_root}")

    for sample_dir in sample_dirs:
        video_id = sample_dir.name
        print(f"\nProcessing sample: {video_id}")

        support_dir = sample_dir / "object_images"
        if not support_dir.is_dir():
            print(f"  Skipping {video_id}: no object_images/ directory")
            results.append({"video_id": video_id, "detections": []})
            continue

        support_paths = sorted(
            [p for p in support_dir.glob("*.jpg")] + [p for p in support_dir.glob("*.png")]
        )
        if len(support_paths) == 0:
            print(f"  Skipping {video_id}: no support images found in {support_dir}")
            results.append({"video_id": video_id, "detections": []})
            continue

        # Find video file (prefer drone_video.mp4, else first .mp4)
        video_path = sample_dir / "drone_video.mp4"
        if not video_path.exists():
            mp4s = list(sample_dir.glob("*.mp4"))
            if not mp4s:
                print(f"  Skipping {video_id}: no .mp4 video found")
                results.append({"video_id": video_id, "detections": []})
                continue
            video_path = mp4s[0]

        print(f"  Using video: {video_path.name}")
        print(f"  #supports: {len(support_paths)}")

        # Optional video output path
        vid_out_path = None
        if save_video:
            vid_out_path = video_out_dir / f"{video_id}_pred.mp4"

        # Run evaluation on this video
        bboxes = eval_single_video(
            model=model,
            support_paths=support_paths,
            video_path=video_path,
            device=device,
            score_threshold=score_threshold,
            resize_short_side=resize_short_side,
            save_video=save_video,
            video_out_path=vid_out_path,
        )

        if len(bboxes) > 0:
            detections = [{"bboxes": bboxes}]
        else:
            detections = []

        results.append(
            {
                "video_id": video_id,
                "detections": detections,
            }
        )

    # Write JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detections JSON to: {output_json}")


# =========================================================
# ARG PARSER
# =========================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate Siamese detector on samples/ and export JSON detections."
    )

    ap.add_argument("--checkpoint", required=True, help="Path to Siamese checkpoint (.pth)")
    ap.add_argument("--samples_root", required=True, help="Root of samples/ (e.g. train/samples)")
    ap.add_argument("--output_json", required=True, help="Path to output JSON file")

    ap.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    ap.add_argument("--score_threshold", type=float, default=0.5,
                    help="Score threshold for including detections in JSON")
    ap.add_argument("--resize_short_side", type=int, default=None,
                    help="Resize short side of frames to this size (optional)")

    ap.add_argument("--save_video", action="store_true",
                    help="If set, save visualization videos for each sample")
    ap.add_argument("--video_out_dir", default="eval_videos",
                    help="Directory to save visualization videos (used if --save_video)")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_eval_over_samples(
        checkpoint=Path(args.checkpoint),
        samples_root=Path(args.samples_root),
        output_json=Path(args.output_json),
        device_name=args.device,
        score_threshold=args.score_threshold,
        save_video=args.save_video,
        video_out_dir=Path(args.video_out_dir) if args.save_video else None,
        resize_short_side=args.resize_short_side,
    )
