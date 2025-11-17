#!/usr/bin/env python3
"""
Fully fixed and rewritten evaluation script for your Cross-Attention Siamese detector.

This version:
- Matches training preprocessing EXACTLY.
- Uses correct denormalization logic.
- Makes Faster R-CNN keep more proposals (higher recall).
- Works with CrossAttentionSimilarityHead producing [bg, fg] logits.
- Draws boxes at multiple thresholds.
- Uses support images to form prototype ONCE per run.
"""

import argparse
from pathlib import Path
from typing import List
import cv2
import torch
import numpy as np
from tqdm import tqdm

from SM2 import build_siamese_model, make_support_proto, denormalize_for_model
from Data2 import IMAGENET_MEAN, IMAGENET_STD


# =========================================================
# SUPPORT IMAGE LOADING
# =========================================================
def load_support_images_as_tensor(paths: List[Path], device: torch.device):
    """Load support images and resize to 256x256 + normalize exactly like training."""
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

        # normalize
        img = (img - mean) / std
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        out.append(t)

    return torch.stack(out, dim=0).to(device)   # (K, C, H, W)


# =========================================================
# QUERY FRAME PREPROCESSING
# =========================================================
def preprocess_frame(frame_bgr, device):
    """Convert BGR → Tensor normalized → then denormalized back to [0,1] to match model input."""
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    norm = (rgb - mean) / std

    # convert to tensor
    t = torch.from_numpy(norm).permute(2, 0, 1).float().to(device)

    # model expects [0,1] denormalized
    t_0_1 = denormalize_for_model(t, device)
    return t_0_1



# =========================================================
# DRAWING
# =========================================================
def draw_threshold_boxes(frame, preds, threshold, color, label_prefix=""):
    boxes = preds["boxes"]
    scores = preds["scores"]

    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]

    count = 0
    for box, s in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label_prefix}{s:.2f}"
        cv2.putText(frame, text, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        count += 1

    return count



# =========================================================
# MAIN EVAL LOGIC
# =========================================================
def eval_video(
        checkpoint,
        support_paths,
        video_in,
        video_out,
        thresholds,
        device_name="cuda",
        max_frames=None,
        resize_short_side=None
):
    # Device ------------------------------------------------
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model --------------------------------------------
    print("Building Siamese model...")
    model = build_siamese_model(agnostic_weights=None)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    print("Loading checkpoint weights...")
    model.load_state_dict(state)   # strict=True
    model.to(device)
    model.eval()

    # Improve inference recall ---------------------------------
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh = 0.7
    model.roi_heads.detections_per_img = 300

    # Load support & compute prototype -------------------------
    support_tensors = load_support_images_as_tensor([Path(p) for p in support_paths], device)
    with torch.no_grad():
        proto = make_support_proto(model, support_tensors, device)
    model.roi_heads.box_predictor.set_prototype(proto)

    # Video IO -------------------------------------------------
    cap = cv2.VideoCapture(video_in if video_in != "0" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_short_side:
        short = min(width, height)
        scale = resize_short_side / float(short)
        width = int(width * scale)
        height = int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    # Colors for thresholds
    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 128, 255)
    ]

    # Process frames ------------------------------------------
    idx = 0
    pbar = tqdm(total=max_frames, desc="Processing frames", disable=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if resize_short_side:
            frame = cv2.resize(frame, (width, height))

        # Prepare model input
        input_tensor = preprocess_frame(frame, device)

        with torch.no_grad():
            out_dict = model([input_tensor])[0]

        # CPU conversion
        preds = {
            "boxes": out_dict["boxes"].cpu().numpy(),
            "scores": out_dict["scores"].cpu().numpy(),
            "labels": out_dict["labels"].cpu().numpy(),
        }

        # Keep only label == 1 (matching class)
        keep = preds["labels"] == 1
        match_preds = {
            "boxes": preds["boxes"][keep],
            "scores": preds["scores"][keep]
        }

        # Drawing
        frame_draw = frame.copy()
        for i, thr in enumerate(thresholds):
            c = COLORS[i % len(COLORS)]
            count = draw_threshold_boxes(frame_draw, match_preds, thr, c, label_prefix=f"{thr:.2f}:")
            cv2.putText(frame_draw, f"thr {thr:.2f}: {count}",
                        (10, 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)

        # Frame index
        cv2.putText(frame_draw, f"frame {idx}",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        out.write(frame_draw)
        idx += 1
        pbar.update(1)

        if max_frames and idx >= max_frames:
            break

    pbar.close()
    cap.release()
    out.release()

    print(f"Saved video: {video_out}")



# =========================================================
# ARG PARSER
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--support_dir", required=True)
    ap.add_argument("--video_in", required=True)
    ap.add_argument("--video_out", required=True)

    ap.add_argument("--score_thresholds", default="0.2,0.4,0.6")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--resize_short_side", type=int, default=None)

    return ap.parse_args()



if __name__ == "__main__":
    args = parse_args()

    support_paths = list(Path(args.support_dir).glob("*"))
    if len(support_paths) == 0:
        raise RuntimeError("No support images found in support_dir")

    thresholds = [float(x) for x in args.score_thresholds.split(",")]

    eval_video(
        checkpoint=args.checkpoint,
        support_paths=[str(p) for p in support_paths],
        video_in=args.video_in,
        video_out=args.video_out,
        thresholds=thresholds,
        device_name=args.device,
        max_frames=args.max_frames,
        resize_short_side=args.resize_short_side
    )
