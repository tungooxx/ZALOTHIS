#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided ByteTrack with always-on visualization + debug HUD.

What's new in v2 (per request):
- **Unified pre-segmentation pipeline** with 3 modes: `off`, `grabcut`, `yolo`.
- **Segmentation can run independently** of color-gating/tracking using `--seg_only`.
- **Exemplar pre-segmentation**:
  * Use `--presegment_exemplars` to mask exemplars and save to `outdir/exemplars_segmented/`.
  * Use `--use_preseg_exemplars` to embed **only** the pre-segmented exemplars.
- **Yolo-seg support** via separate `--seg_weights` (falls back to `--weights` if omitted).
- **Always-available segmentation debug outputs** (masks, overlays, raw + masked object crops), not tied to color gate.
- Backward compatible flags retained.

Color legend (unchanged):
  • Green  = kept (passes similarity + color gate)
  • Red    = failed similarity threshold
  • Orange = failed color gate (when enabled)

"""

import os, glob, json, time, argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
from ultralytics import YOLO


# ---------- helpers ----------
# ---------- color profile (Lab + HSV) ----------
class ColorProfile:
    def __init__(self, exemplar_dir, hsv_bins=(36, 32), resize=None,
                 seg_mode='off', ex_inset=0.06, gc_iter=2):
        self.hsv_bins = hsv_bins
        labs, hists = [], []
        paths = sorted(glob.glob(os.path.join(exemplar_dir, "*")))
        for p in paths[:3]:
            img = cv2.imread(p)
            if img is None:
                continue
            if resize is not None:
                img = cv2.resize(img, resize)
            # Build a foreground mask for exemplar to suppress background
            H, W = img.shape[:2]
            if seg_mode == 'off':
                ex_mask = None
            else:
                # rectangle covering the central region as probable foreground
                mx1 = int(W * ex_inset)
                my1 = int(H * ex_inset)
                mx2 = int(W * (1 - ex_inset))
                my2 = int(H * (1 - ex_inset))
                rect = (mx1, my1, max(1, mx2 - mx1), max(1, my2 - my1))
                # GrabCut init with rectangle
                mask = np.zeros((H, W), np.uint8)
                bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
                try:
                    cv2.grabCut(img, mask, rect, bg, fg, gc_iter, cv2.GC_INIT_WITH_RECT)
                    ex_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                except Exception:
                    ex_mask = None

            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            if ex_mask is not None:
                sel = lab_img[ex_mask > 0].reshape(-1, 3)
                if sel.size == 0:
                    sel = lab_img.reshape(-1, 3)
            else:
                sel = lab_img.reshape(-1, 3)
            labs.append(sel.mean(axis=0))

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], ex_mask, self.hsv_bins, [0, 180, 0, 256])
            h = cv2.normalize(h, None).flatten()
            hists.append(h)

        if not labs or not hists:
            raise FileNotFoundError(f"No valid exemplar images to build color profile in {exemplar_dir}")

        self.lab_mean = np.mean(np.stack(labs, 0), axis=0)  # (L,a,b)
        hh = np.mean(np.stack(hists, 0), axis=0)
        self.hsv_hist = hh / (np.linalg.norm(hh) + 1e-8)  # L2-normalized

    def color_metrics(self, crop_bgr, mask=None):
        """
        Returns:
          dE76: Euclidean distance in Lab (lower is better; 0 = identical)
          hsv_corr: cosine similarity of HSV histograms (higher is better; 1 = identical)
        """
        if crop_bgr.size == 0:
            return float("inf"), 0.0
        lab_img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        if mask is not None:
            sel = lab_img[mask > 0].reshape(-1, 3)
            if sel.size == 0:
                sel = lab_img.reshape(-1, 3)
        else:
            sel = lab_img.reshape(-1, 3)
        dE76 = float(np.linalg.norm(sel.mean(axis=0) - self.lab_mean))

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], mask, self.hsv_bins, [0, 180, 0, 256])
        h = cv2.normalize(h, None).flatten()
        h = h / (np.linalg.norm(h) + 1e-8)
        hsv_corr = float(np.dot(h, self.hsv_hist))  # cosine sim
        return dE76, hsv_corr


def infer_query_label(object_images_path: str) -> str:
    p = os.path.abspath(object_images_path.rstrip(os.sep))
    parent = os.path.basename(os.path.dirname(p))
    return parent if parent else "target"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_video_reader(src: str):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise AssertionError(f"Cannot open video: {src}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    return cap, W, H, FPS, N


def draw_box(img, box, text, color=(0, 255, 0), max_width_px=None, sep=' | '):
    """Draw rectangle + wrapped, multi-line label that stays in-frame.
    - Splits on `sep` tokens (default ' | ').
    - Wraps lines to `max_width_px` (defaults to ~box_width and 45% of frame width).
    - Places label above the box when possible; otherwise below.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    H, W = img.shape[:2]
    if max_width_px is None:
        box_w = max(1, x2 - x1)
        max_width_px = max(160, min(int(0.45 * W), int(1.2 * box_w)))

    tokens = [t.strip() for t in str(text).split(sep) if t.strip()]
    if not tokens:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    line_gap = 6
    lines = []
    cur = ''

    def line_width(s: str) -> int:
        (tw, _), _ = cv2.getTextSize(s, font, scale, thickness)
        return tw

    for tok in tokens:
        candidate = tok if cur == '' else (cur + sep + tok)
        if line_width(candidate) <= max_width_px:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)

    sizes = [cv2.getTextSize(ln, font, scale, thickness)[0] for ln in lines]
    max_w = max(w for (w, h) in sizes)
    line_h = max(h for (w, h) in sizes)
    block_w = max_w + 10
    block_h = len(lines) * (line_h + line_gap) + 6

    bx = x1
    by_top = y1 - 8 - block_h
    y_top = by_top if by_top >= 0 else (y2 + 8)
    if y_top + block_h > H:
        y_top = max(0, H - block_h - 2)

    x_left = min(max(2, bx), max(2, W - block_w - 2))

    bg_color = (0, 0, 0)
    alpha = 0.6
    x0, y0 = x_left, y_top
    x1b, y1b = x_left + block_w, y_top + block_h
    overlay_img = img.copy()
    cv2.rectangle(overlay_img, (x0, y0), (x1b, y1b), bg_color, -1)
    cv2.addWeighted(overlay_img, alpha, img, 1 - alpha, 0, img)

    y = y0 + line_h + 2
    for ln in lines:
        cv2.putText(img, ln, (x_left + 5, y), font, scale, color, thickness)
        y += line_h + line_gap


def draw_hud(img, args):
    """Top-left overlay with current thresholds / gate mode."""
    lines = [f"sim_thr={getattr(args,'sim_thr',0.28):.2f} | color_gate={getattr(args,'color_gate','off')}"]
    if getattr(args,'color_gate','off') in ("lab", "both"):
        lines.append(f"lab_thr={getattr(args,'lab_thr',20.0):.1f}")
    if getattr(args,'color_gate','off') in ("hsv", "both"):
        lines.append(f"hsv_corr_thr={getattr(args,'hsv_corr_thr',0.60):.2f}")
    if getattr(args,'color_gate','off') != "off":
        lines.append(f"inset={getattr(args,'color_inset',0.20):.2f}")
    if getattr(args, 'seg_only', False):
        lines.append("[SEGMENTATION ONLY]")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, pad = 0.6, 2, 6
    x, y = 8, 22
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, font, scale, thick)
        cv2.rectangle(img, (x - 4, y + 4), (x + tw + pad, y - th - 6), (0, 0, 0), -1)
        cv2.putText(img, ln, (x, y), font, scale, (255, 255, 255), thick)
        y += th + 10


# ---------- dataset chooser ----------

def discover_datasets(root_dir: str, mp4_glob: str = "*.mp4"):
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--root not found or not a directory: {root}")
    candidates = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        oi_dir = sub / "object_images"
        if not oi_dir.is_dir():
            continue
        imgs = sorted(glob.glob(str(oi_dir / "*")))
        if not imgs:
            continue
        mp4s = sorted(glob.glob(str(sub / mp4_glob)))
        if not mp4s:
            continue
        candidates.append({
            "name": sub.name,
            "base": str(sub),
            "mp4_path": mp4s[0],  # pick first if multiple
            "object_images": str(oi_dir)
        })
    return candidates


def pretty_print_datasets(datasets):
    print("\n[Available datasets]")
    for i, d in enumerate(datasets, 1):
        print(f" {i}. {d['name']}")
        print(f"    mp4: {d['mp4_path']}")
        print(f"    object_images: {d['object_images']}")
    print("")


# ---------- Exemplar embedder ----------
class ExemplarEmbedder:
    """
    Loads an OpenCLIP image encoder.
    Defaults to MobileCLIP if available; falls back to ViT-B/32 (openai).
    """

    def __init__(self, model_name="MobileCLIP-S1", pretrained=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def _create(m, p):
            return open_clip.create_model_and_transforms(m, pretrained=p, device=self.device)

        try:
            if pretrained is None:
                avail = dict(open_clip.list_pretrained())  # {model_name: [tags]}
                key = next((k for k in avail if k.lower() == model_name.lower()), None)
                if key and avail.get(key):
                    pretrained = "datacompdr"
                    model_name = key
                else:
                    for cand in ["MobileCLIP2-S4", "MobileCLIP-S1", "ViT-B-32"]:
                        if cand in avail and avail[cand]:
                            model_name, pretrained = cand, avail[cand][0]
                            break
            self.model, _, self.preprocess = _create(model_name, pretrained)
            print(f"[Embedder] Loaded {model_name} ({pretrained}) on {self.device}")
        except Exception as e:
            print(f"[Embedder] Failed {model_name} ({pretrained}): {e}")
            print("[Embedder] Falling back to ViT-B-32 (openai).")
            self.model, _, self.preprocess = _create("ViT-B-32", "openai")
            print(f"[Embedder] Loaded ViT-B-32 (openai) on {self.device}")

        self.model.eval()

    @torch.no_grad()
    def embed_pils(self, pil_list):
        ims = [self.preprocess(im).unsqueeze(0) for im in pil_list]
        batch = torch.cat(ims, dim=0).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
            feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def embed_dir_mean(self, d):
        paths = sorted(glob.glob(os.path.join(d, "*")))
        imgs = []
        for p in paths[:3]:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        if not imgs:
            raise FileNotFoundError(f"No exemplar images in {d}")
        feats = self.embed_pils(imgs)
        mean = feats.mean(dim=0, keepdim=True)
        return mean / mean.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def sim_of_crop_bgr(self, crop_bgr, exemplar_feat):
        if crop_bgr.size == 0:
            return -1.0
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        f = self.embed_pils([pil])
        return float((f @ exemplar_feat.t()).squeeze().item())


# ---------- simple smoother ----------
class BoxEMASmoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.buf = {}

    def update(self, tid, box):
        b = np.asarray(box, np.float32)
        if tid not in self.buf:
            self.buf[tid] = b
        else:
            self.buf[tid] = self.alpha * b + (1 - self.alpha) * self.buf[tid]
        return self.buf[tid]


def save_context_crop(frame_bgr, coords, out_path):
    cx1, cy1, cx2, cy2 = coords
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    ensure_dir(Path(out_path).parent)
    cv2.imwrite(out_path, crop)


def inset_rect(x1, y1, x2, y2, inset_ratio, W, H):
    if inset_ratio <= 0.0:
        return int(x1), int(y1), int(x2), int(y2)
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    ix1 = int(round(x1 + inset_ratio * w))
    iy1 = int(round(y1 + inset_ratio * h))
    ix2 = int(round(x2 - inset_ratio * w))
    iy2 = int(round(y2 - inset_ratio * h))
    ix1 = max(0, min(ix1, W - 2))
    iy1 = max(0, min(iy1, H - 2))
    ix2 = max(ix1 + 1, min(ix2, W - 1))
    iy2 = max(iy1 + 1, min(iy2, H - 1))
    return ix1, iy1, ix2, iy2


# ---------- segmentation helpers ----------

def grabcut_rect_mask(bgr, rect, iters=2):
    H, W = bgr.shape[:2]
    x, y, w, h = rect
    x = max(0, min(x, W - 2)); y = max(0, min(y, H - 2))
    w = max(1, min(w, W - x - 1)); h = max(1, min(h, H - y - 1))
    mask = np.zeros((H, W), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x, y, w, h), bg, fg, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def get_seg_mask(frame, x1, y1, x2, y2, mode='grabcut', inset=0.1, iters=2, result=None, det_index=None):
    H, W = frame.shape[:2]
    if mode == 'yolo' and result is not None and hasattr(result, 'masks') and result.masks is not None:
        try:
            m = result.masks.data[det_index].cpu().numpy()
            if m.dtype != np.uint8:
                m = (m > 0.5).astype(np.uint8) * 255
            if m.shape[-2:] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = np.zeros((H, W), np.uint8)
            xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
            mask[yi1:yi2, xi1:xi2] = m[yi1:yi2, xi1:xi2]
            return mask
        except Exception:
            pass
    if mode == 'off':
        mask = np.zeros((H, W), np.uint8)
        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
        mask[yi1:yi2, xi1:xi2] = 255
        return mask
    rx1 = int(x1 + inset * (x2 - x1))
    ry1 = int(y1 + inset * (y2 - y1))
    rx2 = int(x2 - inset * (x2 - x1))
    ry2 = int(y2 - inset * (y2 - y1))
    rect = (rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))
    return grabcut_rect_mask(frame, rect, iters)


def apply_mask_to_crop(crop_bgr, mask_crop, transparent=False):
    """Return masked crop (RGB) with black or transparent background."""
    if mask_crop is None or crop_bgr.size == 0:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    m = (mask_crop > 0).astype(np.uint8)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if not transparent:
        out = rgb.copy()
        out[m == 0] = 0
        return out
    # RGBA
    a = (m * 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    return rgba


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    # core
    ap.add_argument("--weights", required=True, help="YOLO .pt weights (detection or seg)")
    ap.add_argument("--seg_weights", default=None, help="Optional YOLO-seg .pt for masks; falls back to --weights")
    ap.add_argument("--source", help="Video path (.mp4, .avi, etc.)")
    ap.add_argument("--object_images", help="Folder with exemplar images")
    ap.add_argument("--outdir", default="runs_guided", help="Output directory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--topk", type=int, default=120)
    ap.add_argument("--sim_thr", type=float, default=0.28)
    ap.add_argument("--clip_model", default="MobileCLIP-S1")
    ap.add_argument("--clip_tag", default="datacompdr", help="Override pretrained tag if you want")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--display_label", default=None)

    # segmentation options (now used broadly)
    ap.add_argument("--seg_mode", choices=["off", "grabcut", "yolo"], default="grabcut",
                    help="How to get object masks: 'off' (box), 'grabcut' (fast), 'yolo' (use seg model masks)")
    ap.add_argument("--seg_grabcut_iter", type=int, default=2,
                    help="GrabCut iterations for per-frame masks")
    ap.add_argument("--seg_inset", type=float, default=0.10,
                    help="Inset ratio for the rectangle used to initialize GrabCut around each detection")
    ap.add_argument("--exemplar_seg_inset", type=float, default=0.06,
                    help="Inset ratio used to mask exemplars (center region is probable foreground)")

    # segmentation-only mode
    ap.add_argument("--seg_only", action="store_true", help="Run segmentation + saving only; skip similarity/color gate")
    ap.add_argument("--seg_transparent", action="store_true", help="Save masked crops as RGBA with transparent bg")

    # dataset chooser for your layout
    ap.add_argument("--root", help="Root containing dataset folders with object_images/ and an mp4 inside each")
    ap.add_argument("--choose", type=int, help="Pick the Nth dataset (1-based) under --root")
    ap.add_argument("--list_datasets", action="store_true", help="List discovered datasets under --root and exit")
    ap.add_argument("--mp4_glob", default="*.mp4", help="Pattern for selecting the video file in each dataset folder")

    # context saving
    ap.add_argument("--save_context", action="store_true", help="Save context JPG crops for kept detections")
    ap.add_argument("--max_context_per_id", type=int, default=25, help="Limit number of context crops per track ID")
    ap.add_argument("--context_pad", type=float, default=0.08, help="Padding ratio around box for context crop")
    ap.add_argument("--context", type=float, default=None, help="Alias for --context_pad (deprecated)")

    # color gate (optional)
    ap.add_argument("--color_gate", choices=["off", "lab", "hsv", "both"], default="off",
                    help="Add color check on top of CLIP sim")
    ap.add_argument("--lab_thr", type=float, default=20.0,
                    help="Max Lab ΔE (CIE76) allowed when color_gate uses Lab/both (lower = stricter)")
    ap.add_argument("--hsv_corr_thr", type=float, default=0.60,
                    help="Min HSV histogram cosine similarity allowed when color_gate uses HSV/both (higher = stricter)")

    # mask saving / debug (always available)
    ap.add_argument("--save_mask", action="store_true", help="Save segmentation masks (ROI) for debug")
    ap.add_argument("--mask_stride", type=int, default=1, help="Save masks every N frames per track")
    ap.add_argument("--mask_for_all_dets", action="store_true",
                    help="Save masks for all detections instead of only kept ones")
    ap.add_argument("--mask_overlay", action="store_true", help="Save colored overlay of mask on ROI for visualization")
    ap.add_argument("--color_inset", type=float, default=0.20,
                    help="Fraction to inset the detection box per side for color metrics")

    # exemplar pre-segmentation controls
    ap.add_argument("--presegment_exemplars", action="store_true", help="Pre-segment exemplars and save outputs")
    ap.add_argument("--use_preseg_exemplars", action="store_true", help="Use only pre-segmented exemplars for embedding")
    ap.add_argument("--preseg_dir", default=None, help="Optional directory to read/write pre-segmented exemplars")

    args = ap.parse_args()

    # Backward-compat: allow --context as alias for --context_pad
    if getattr(args, "context", None) is not None:
        args.context_pad = float(args.context)

    # resolve dataset selection from --root/--choose
    selected = None
    if args.root:
        datasets = discover_datasets(args.root, mp4_glob=args.mp4_glob)
        if args.list_datasets:
            pretty_print_datasets(datasets)
            return
        if args.choose is not None:
            if not datasets:
                raise RuntimeError(f"No datasets found under --root {args.root}")
            if args.choose < 1 or args.choose > len(datasets):
                raise IndexError(f"--choose must be between 1 and {len(datasets)}")
            selected = datasets[args.choose - 1]

    # fill in source/object_images from selection if not explicitly provided
    if selected and (args.source is None and args.object_images is None):
        args.source = selected["mp4_path"]
        args.object_images = selected["object_images"]
        if args.display_label is None:
            args.display_label = selected["name"]

    if not args.source or not args.object_images:
        raise AssertionError("Provide (--source and --object_images) OR (--root and --choose).")

    # IO
    outdir = Path(args.outdir); ensure_dir(outdir)
    out_mp4 = str(outdir / "guided_preview.mp4")
    out_json = str(outdir / "tubes.json")
    ctx_root = outdir / "contexts"
    mask_root = outdir / "segmentation"
    objseg_root = outdir / "seg_objects"  # masked object crops
    ensure_dir(objseg_root)

    # ---------- exemplar pre-segmentation ----------
    preseg_dir = Path(args.preseg_dir) if args.preseg_dir else (outdir / "exemplars_segmented")
    if args.presegment_exemplars:
        ensure_dir(preseg_dir)
        print(f"[PreSeg] Segmenting exemplars from {args.object_images} -> {preseg_dir}")
        for p in sorted(glob.glob(os.path.join(args.object_images, "*"))):
            img = cv2.imread(p)
            if img is None:
                continue
            H, W = img.shape[:2]
            # Build a mask per seg_mode for exemplar
            if args.seg_mode == 'yolo':
                # Use rectangle grabcut fallback for exemplars (no result object)
                mask = grabcut_rect_mask(img, (
                    int(W*args.exemplar_seg_inset), int(H*args.exemplar_seg_inset),
                    int(W*(1-2*args.exemplar_seg_inset)), int(H*(1-2*args.exemplar_seg_inset))
                ), iters=args.seg_grabcut_iter)
            elif args.seg_mode == 'grabcut':
                mask = grabcut_rect_mask(img, (
                    int(W*args.exemplar_seg_inset), int(H*args.exemplar_seg_inset),
                    int(W*(1-2*args.exemplar_seg_inset)), int(H*(1-2*args.exemplar_seg_inset))
                ), iters=args.seg_grabcut_iter)
            else:  # off => box mask
                mask = np.ones((H, W), np.uint8) * 255
            # apply
            out_rgba = apply_mask_to_crop(img, mask, transparent=args.seg_transparent)
            out_name = preseg_dir / (Path(p).stem + (".png" if args.seg_transparent else "_masked.jpg"))
            cv2.imwrite(str(out_name), cv2.cvtColor(out_rgba, cv2.COLOR_RGBA2BGRA) if out_rgba.shape[-1]==4 else cv2.cvtColor(out_rgba, cv2.COLOR_RGB2BGR))
        print(f"[PreSeg] Done. Wrote to {preseg_dir}")

    # If requested, replace exemplar dir with preseg_dir for embedding
    exemplar_dir_for_embed = str(preseg_dir if args.use_preseg_exemplars else args.object_images)

    # Optional color profile
    color_profile = None
    if args.color_gate != "off":
        color_profile = ColorProfile(
            exemplar_dir_for_embed,
            seg_mode=args.seg_mode,
            ex_inset=args.exemplar_seg_inset,
            gc_iter=args.seg_grabcut_iter,
        )
        print(f"[ColorGate] Enabled: {args.color_gate}  lab_thr={args.lab_thr}  hsv_corr_thr={args.hsv_corr_thr}")

    # label text
    query_label = args.display_label or infer_query_label(args.object_images)
    print(f"[Info] Query label: {query_label}")

    # models
    print("[Load] YOLO weights (detect/track):", args.weights)
    model = YOLO(args.weights)

    # separate seg model if provided
    seg_model = None
    if args.seg_mode == 'yolo':
        seg_w = args.seg_weights if args.seg_weights is not None else args.weights
        print("[Load] YOLO (seg) weights:", seg_w)
        seg_model = YOLO(seg_w)

    embedder = ExemplarEmbedder(model_name=args.clip_model, pretrained=args.clip_tag)
    exemplar_feat = embedder.embed_dir_mean(exemplar_dir_for_embed)

    # video
    cap, W, H, FPS, N = load_video_reader(args.source)
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4, fourcc, FPS, (W, H))

    # tracker stream (we still use .track for convenience; in seg_only we can skip gates)
    stream = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=0 if torch.cuda.is_available() else "cpu",
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        show=False,
        verbose=False
    )

    smoother = BoxEMASmoother(alpha=0.35)
    tubes = defaultdict(list)
    t0 = time.time()
    frame_idx = -1
    pbar = tqdm(total=N, desc="Guided ByteTrack v2 (.pt)")
    saved_counts = defaultdict(int)

    for result in stream:
        frame_idx += 1
        now = time.time() - t0
        frame = result.orig_img
        if frame is None:
            pbar.update(1); continue

        draw_hud(frame, args)

        if result.boxes is None or len(result.boxes) == 0:
            if writer: writer.write(frame)
            pbar.update(1); continue

        ids = (result.boxes.id.cpu().numpy().astype(int)
               if hasattr(result.boxes, "id") and result.boxes.id is not None else None)
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()

        if ids is None or len(ids) == 0:
            ids = np.full((len(xyxy),), -1, dtype=int)

        Hf, Wf = frame.shape[:2]
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-areas)[:args.topk]

        # If using YOLO seg, re-run seg_model.predict on this frame once for masks (faster than per-detection)
        seg_result_for_frame = None
        if args.seg_mode == 'yolo' and seg_model is not None:
            try:
                seg_preds = list(seg_model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False))
                seg_result_for_frame = seg_preds[0] if len(seg_preds)>0 else None
            except Exception:
                seg_result_for_frame = None

        for i in order:
            tid = int(ids[i])
            x1, y1, x2, y2 = xyxy[i]
            pad_x = args.context_pad * (x2 - x1)
            pad_y = args.context_pad * (y2 - y1)
            cx1 = max(0, int(x1 - pad_x)); cy1 = max(0, int(y1 - pad_y))
            cx2 = min(Wf - 1, int(x2 + pad_x)); cy2 = min(Hf - 1, int(y2 + pad_y))
            crop = frame[cy1:cy2, cx1:cx2]

            # Segmentation mask (frame space) always available if seg_mode != off
            roi_mask = None
            if args.seg_mode != 'off':
                seg_result_use = seg_result_for_frame if args.seg_mode=='yolo' else result
                roi_full = get_seg_mask(frame, x1, y1, x2, y2,
                                        mode=args.seg_mode,
                                        inset=args.seg_inset,
                                        iters=args.seg_grabcut_iter,
                                        result=seg_result_use,
                                        det_index=i)
                roi_mask = roi_full[cy1:cy2, cx1:cx2] if roi_full is not None else None

            # Save segmentation debug outputs (independent toggle)
            if args.save_mask and ( (args.mask_for_all_dets) or (tid>=0) ) and ((frame_idx % max(1, args.mask_stride))==0):
                ensure_dir(mask_root / f"{query_label}" / f"id_{tid}")
                mask_path = mask_root / f"{query_label}" / f"id_{tid}" / f"frame_{frame_idx:06d}_mask.png"
                if roi_mask is None:
                    tmp = np.zeros((max(1, cy2 - cy1), max(1, cx2 - cx1)), np.uint8)
                    cv2.imwrite(str(mask_path), tmp)
                else:
                    cv2.imwrite(str(mask_path), roi_mask)
                if args.mask_overlay:
                    roi = frame[cy1:cy2, cx1:cx2].copy()
                    if roi_mask is not None:
                        color_mask = np.zeros_like(roi)
                        color_mask[roi_mask > 0] = (0, 255, 0)
                        cv2.addWeighted(color_mask, 0.35, roi, 0.65, 0, roi)
                    ov_path = mask_root / f"{query_label}" / f"id_{tid}" / f"frame_{frame_idx:06d}_overlay.jpg"
                    cv2.imwrite(str(ov_path), roi)

            # Save object crops (raw + masked) for debug
            ensure_dir(objseg_root / f"{query_label}" / f"id_{tid}")
            raw_path = objseg_root / f"{query_label}" / f"id_{tid}" / f"frame_{frame_idx:06d}_raw.jpg"
            cv2.imwrite(str(raw_path), crop)
            if roi_mask is not None:
                masked = apply_mask_to_crop(crop, roi_mask, transparent=args.seg_transparent)
                ext = ".png" if args.seg_transparent else ".jpg"
                masked_path = (
                        objseg_root / f"{query_label}" / f"id_{tid}" / f"frame_{frame_idx:06d}_masked{ext}"
                )
                if masked.ndim==3 and masked.shape[-1]==4:
                    cv2.imwrite(str(masked_path), cv2.cvtColor(masked, cv2.COLOR_RGBA2BGRA))
                else:
                    cv2.imwrite(str(masked_path), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

            # If segmentation-only, we don't do similarity/gates/tubes; just draw and continue
            if args.seg_only:
                draw_box(frame, (x1, y1, x2, y2), f"{query_label} | id{tid} | conf {conf[i]:.2f} | SEG_ONLY", color=(0,255,255))
                continue

            # Similarity score
            sim = ExemplarEmbedder.sim_of_crop_bgr(embedder, crop, exemplar_feat)

            # Color metrics (if enabled)
            dE76, hsv_corr = (float("nan"), float("nan"))
            color_ok = True
            if color_profile is not None and args.color_gate != "off":
                if roi_mask is None:
                    # build a focused box region for color
                    fx1, fy1, fx2, fy2 = inset_rect(x1, y1, x2, y2, args.color_inset, Wf, Hf)
                    roi_mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                    roi_mask[fy1:fy2, fx1:fx2] = 255
                    roi_mask = roi_mask[cy1:cy2, cx1:cx2]
                dE76, hsv_corr = color_profile.color_metrics(crop, mask=roi_mask)
                if args.color_gate == "lab":
                    color_ok = (dE76 <= args.lab_thr)
                elif args.color_gate == "hsv":
                    color_ok = (hsv_corr >= args.hsv_corr_thr)
                elif args.color_gate == "both":
                    color_ok = (dE76 <= args.lab_thr) and (hsv_corr >= args.hsv_corr_thr)

            sim_ok = (sim >= args.sim_thr)
            keep = sim_ok and color_ok

            # Smoothing if we have a valid track ID; otherwise draw raw box.
            draw_box_coords = xyxy[i]
            if tid >= 0:
                draw_box_coords = smoother.update(tid, xyxy[i])

            status = "OK" if keep else ("SIM_FAIL" if not sim_ok else "COLOR_FAIL")
            overlay = f"{query_label} | id{tid} | conf {conf[i]:.2f} | sim {sim:.2f} [{status}]"
            if args.color_gate != "off":
                overlay += f" | dE {dE76:.1f} | hsv {hsv_corr:.2f}"
            col = (0, 255, 0) if keep else ((0, 0, 255) if not sim_ok else (0, 165, 255))
            draw_box(frame, draw_box_coords, overlay, color=col)

            # Persist only kept detections with a valid track id
            if keep and tid >= 0:
                tubes[tid].append({
                    "t": float(now), "f": int(frame_idx),
                    "box": [float(v) for v in np.asarray(draw_box_coords, dtype=float).tolist()],
                    "cls": 0, "conf": float(conf[i]),
                    "sim": float(sim)
                })
                if args.save_context and (saved_counts[tid] < args.max_context_per_id):
                    out_path = ctx_root / f"{query_label}" / f"id_{tid}" / f"frame_{frame_idx:06d}_sim_{sim:.2f}.jpg"
                    ensure_dir(out_path.parent)
                    save_context_crop(frame, (cx1, cy1, cx2, cy2), str(out_path))
                    saved_counts[tid] += 1

        if writer:
            writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer: writer.release()

    # jsonify defaultdict keys
    tubes_jsonable = {str(k): v for k, v in tubes.items()}
    with open(out_json, "w") as f:
        json.dump({"fps": float(FPS), "query_label": query_label, "tubes": tubes_jsonable}, f, indent=2)
    print("[DONE] Saved tubes:", out_json)
    if writer:
        print("[DONE] Saved video:", out_mp4)
    if args.save_context:
        print("[DONE] Saved context crops under:", str(ctx_root))
    print("[DONE] Saved masks under:", str(mask_root))
    print("[DONE] Saved (raw/masked) object crops under:", str(objseg_root))


if __name__ == "__main__":
    main()
