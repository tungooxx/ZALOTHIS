#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided ByteTrack + segmentation + CLIP gate (batched) — with deep debug.

- YOLO(.pt) + optional ByteTrack for det/track
- Segmentation: off | grabcut | yolo
- CLIP gating only (color gate removed)
- Batched CLIP embeddings for all crops per frame
- Masked embedding option to reduce background bias
- Rich debug: CSV, npy features, montages, histograms, exemplar diagnostics

Run (predict, no tracking — default):
python guided_track_pt_v2_nocolor_clipbatch.py \
  --weights best_yolo11_960.pt \
  --source /home/chucky/PycharmProjects/ZALOTHIS/public_test/samples/BlackBox_0/drone_video.mp4 \
  --object_images /home/chucky/PycharmProjects/ZALOTHIS/public_test/samples/BlackBox_0/object_images \
  --infer predict --save_video --imgsz 640 --conf 0.2 --iou 0.45 \
  --topk 120 --sim_thr 0.28 \
  --seg_mode grabcut --seg_grabcut_iter 2 --seg_inset 0.10 \
  --clip_batch 64 --clip_use_mask \
  --exemplar_mode per \
  --debug_dir runs_guided/debug_bb0 --dump_feats --viz_topk 12 --save_mask --mask_overlay

Run (enable tracking with ByteTrack):
python guided_track_pt_v2_nocolor_clipbatch.py \
  --weights best_yolo11_960.pt \
  --source /path/to/video.mp4 \
  --object_images /path/to/object_images \
  --infer track --tracker_cfg bytetrack.yaml
"""

import os, glob, json, time, argparse, csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

import torch
import open_clip
from ultralytics import YOLO

from tqdm import tqdm

# -------------------- small utils --------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def infer_query_label(object_images_path: str) -> str:
    p = os.path.abspath(object_images_path.rstrip(os.sep))
    parent = os.path.basename(os.path.dirname(p))
    return parent if parent else "target"

def load_video_reader(src: str):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise AssertionError(f"Cannot open video: {src}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    return cap, W, H, FPS, N

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

def draw_box(img, box, text, color=(0, 255, 0)):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_hud(img, sim_thr, seg_mode, infer_mode):
    txt = f"sim_thr={sim_thr:.2f} | seg={seg_mode} | infer={infer_mode}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (4, 4), (10 + tw, 10 + th), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def apply_mask_to_crop(crop_bgr, mask_crop, transparent=False):
    if mask_crop is None or crop_bgr.size == 0:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    m = (mask_crop > 0).astype(np.uint8)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if not transparent:
        out = rgb.copy()
        out[m == 0] = 0
        return out
    a = (m * 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    return rgba

# -------------------- segmentation --------------------

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

# -------------------- smoother --------------------

class BoxEMASmoother:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.buf = {}
    def update(self, tid, box):
        b = np.asarray(box, np.float32)
        self.buf[tid] = b if tid not in self.buf else self.alpha * b + (1 - self.alpha) * self.buf[tid]
        return self.buf[tid]

# -------------------- CLIP embedder (batched) --------------------

class ExemplarEmbedder:
    """
    Batched OpenCLIP image encoder with (optional) masked crops.
    exemplar_mode:
      - 'mean' : average(normalized feats of exemplar images) -> 1xD
      - 'per'  : keep each exemplar feat and take max similarity across them
    """
    def __init__(self, model_name="MobileCLIP-S1", pretrained=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        def _create(m, p):
            return open_clip.create_model_and_transforms(m, pretrained=p, device=self.device)

        try:
            if pretrained is None:
                avail = dict(open_clip.list_pretrained())
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
            print(f"[Embedder] {model_name} ({pretrained}) on {self.device}")
        except Exception as e:
            print(f"[Embedder] Failed {model_name} ({pretrained}): {e}")
            print("[Embedder] Falling back to ViT-B-32 (openai).")
            self.model, _, self.preprocess = _create("ViT-B-32", "openai")
            print(f"[Embedder] ViT-B-32 (openai) on {self.device}")
        self.model.eval()

    def _pil_batch_to_feats(self, pil_list, batch_size=64):
        feats = []
        for i in range(0, len(pil_list), batch_size):
            chunk = pil_list[i:i+batch_size]
            ims = [self.preprocess(im).unsqueeze(0) for im in chunk]
            batch = torch.cat(ims, dim=0).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device == "cuda")):
                f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        return torch.cat(feats, dim=0) if feats else torch.empty(0, self.model.visual.output_dim, device=self.device)

    @torch.no_grad()
    def embed_exemplar_dir(self, d, mode="mean"):
        paths = sorted(glob.glob(os.path.join(d, "*")))
        imgs = []
        for p in paths[:8]:  # up to 8 exemplars
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        if not imgs:
            raise FileNotFoundError(f"No exemplar images in {d}")
        feats = self._pil_batch_to_feats(imgs, batch_size=8)
        if mode == "mean":
            m = feats.mean(dim=0, keepdim=True)
            m = m / m.norm(dim=-1, keepdim=True)
            return {"mode":"mean", "feats": m}  # (1,D)
        else:
            return {"mode":"per", "feats": feats}  # (E,D)

    @torch.no_grad()
    def sims_for_crops(self, crops_bgr, masks=None, use_mask=True, batch_size=64, exemplar_pack=None):
        pil_list = []
        for idx, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0:
                pil_list.append(Image.fromarray(np.zeros((2,2,3), np.uint8)))
                continue
            if use_mask and masks is not None and masks[idx] is not None:
                rgb = apply_mask_to_crop(crop, masks[idx], transparent=False)
            else:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_list.append(Image.fromarray(rgb))
        crop_feats = self._pil_batch_to_feats(pil_list, batch_size=batch_size)  # (N,D)

        if exemplar_pack["mode"] == "mean":
            E = exemplar_pack["feats"]  # (1,D)
            sims = (crop_feats @ E.t()).squeeze(1)  # (N,)
        else:
            E = exemplar_pack["feats"]  # (K,D)
            sims_all = crop_feats @ E.t()           # (N,K)
            sims, _ = sims_all.max(dim=1)           # (N,)
        return sims, crop_feats

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    # IO / core
    ap.add_argument("--weights", required=True, help="YOLO .pt weights (detect or seg)")
    ap.add_argument("--seg_weights", default=None, help="Optional YOLO-seg .pt for masks; falls back to --weights")
    ap.add_argument("--source", required=True, help="Video path")
    ap.add_argument("--object_images", required=True, help="Folder with exemplar images")
    ap.add_argument("--outdir", default="runs_guided", help="Output directory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--topk", type=int, default=120)
    ap.add_argument("--sim_thr", type=float, default=0.28)
    ap.add_argument("--save_video", action="store_true")

    # NEW: inference mode (predict vs track)
    ap.add_argument("--infer", choices=["predict", "track"], default="predict",
                    help="Use YOLO predict (no IDs) or track (ByteTrack)")
    ap.add_argument("--tracker_cfg", default="bytetrack.yaml",
                    help="Tracker config yaml when --infer=track")

    # segmentation
    ap.add_argument("--seg_mode", choices=["off", "grabcut", "yolo"], default="grabcut")
    ap.add_argument("--seg_grabcut_iter", type=int, default=2)
    ap.add_argument("--seg_inset", type=float, default=0.10)
    ap.add_argument("--seg_only", action="store_true")
    ap.add_argument("--seg_transparent", action="store_true")
    ap.add_argument("--save_mask", action="store_true")
    ap.add_argument("--mask_stride", type=int, default=1)
    ap.add_argument("--mask_for_all_dets", action="store_true")
    ap.add_argument("--mask_overlay", action="store_true")

    # context/crops
    ap.add_argument("--context_pad", type=float, default=0.08)
    ap.add_argument("--save_context", action="store_true")
    ap.add_argument("--max_context_per_id", type=int, default=25)

    # CLIP
    ap.add_argument("--clip_model", default="MobileCLIP-S1")
    ap.add_argument("--clip_tag", default="datacompdr")
    ap.add_argument("--clip_batch", type=int, default=64)
    ap.add_argument("--clip_use_mask", action="store_true")
    ap.add_argument("--exemplar_mode", choices=["mean","per"], default="mean")

    # debug
    ap.add_argument("--debug_dir", default=None, help="Directory to dump debug artifacts")
    ap.add_argument("--dump_feats", action="store_true", help="Save npy features for crops/exemplars")
    ap.add_argument("--viz_topk", type=int, default=0, help="Create a top-K montage per frame (0=off)")
    ap.add_argument("--histogram", action="store_true", help="Save per-frame sim histogram as PNG")

    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir: ensure_dir(debug_dir)

    out_mp4 = str(outdir / "guided_preview.mp4")
    out_json = str(outdir / "tubes.json")
    ctx_root = outdir / "contexts"
    mask_root = outdir / "segmentation"
    objseg_root = outdir / "seg_objects"
    ensure_dir(objseg_root)

    # label
    query_label = infer_query_label(args.object_images)
    print(f"[Info] Query label: {query_label}")

    # models
    print("[Load] YOLO det/track:", args.weights)
    model = YOLO(args.weights)
    seg_model = None
    if args.seg_mode == 'yolo':
        seg_w = args.seg_weights if args.seg_weights else args.weights
        print("[Load] YOLO (seg):", seg_w)
        seg_model = YOLO(seg_w)

    # embedder + exemplars
    embedder = ExemplarEmbedder(model_name=args.clip_model, pretrained=args.clip_tag)
    exemplar_pack = embedder.embed_exemplar_dir(args.object_images, mode=args.exemplar_mode)
    if args.dump_feats and debug_dir is not None:
        np.save(debug_dir / "exemplar_feats.npy", exemplar_pack["feats"].detach().cpu().numpy())

    # video
    cap, W, H, FPS, N = load_video_reader(args.source)
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_mp4, fourcc, FPS, (W, H))

    # -------------------- inference stream --------------------
    if args.infer == "track":
        print("[Infer] Using TRACK mode (ByteTrack)")
        stream = model.track(
            source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=0 if torch.cuda.is_available() else "cpu",
            stream=True, persist=True, tracker=args.tracker_cfg, show=False, verbose=False
        )
    else:
        print("[Infer] Using PREDICT mode (no tracking)")
        stream = model.predict(
            source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=0 if torch.cuda.is_available() else "cpu",
            stream=True, verbose=False
        )

    smoother = BoxEMASmoother(alpha=0.35)
    tubes = defaultdict(list)

    # debug CSV
    csv_writer = None
    csv_path = None
    if debug_dir:
        csv_path = debug_dir / "per_frame_dets.csv"
        csv_writer = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_writer)
        csv_w.writerow(["frame","tid","x1","y1","x2","y2","area","mask_fg","sim","keep","reason"])

    def write_csv_row(row):
        if debug_dir:
            csv.writer(csv_writer).writerow(row)

    # helper: montage
    def save_montage(crops, sims, path, cols=6, pad=4):
        if not crops: return
        Hs, Ws = [], []
        thumbs = []
        for c in crops:
            if c is None or c.size == 0: c = np.zeros((32,32,3), np.uint8)
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            thumbs.append(rgb)
            Hs.append(rgb.shape[0]); Ws.append(rgb.shape[1])
        h = int(np.median(Hs)); w = int(np.median(Ws))
        thumbs = [cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA) for t in thumbs]
        rows = (len(thumbs) + cols - 1) // cols
        canvas = np.zeros((rows*h + (rows+1)*pad, cols*w + (cols+1)*pad, 3), np.uint8)
        canvas[:] = 20
        for idx, t in enumerate(thumbs):
            r = idx // cols; c = idx % cols
            y = pad + r*(h+pad); x = pad + c*(w+pad)
            canvas[y:y+h, x:x+w] = t
            cv2.putText(canvas, f"{sims[idx]:.2f}", (x+4, y+h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    # progress
    frame_idx = -1
    pbar = tqdm(total=N, desc="Guided (seg+CLIP)")

    for result in stream:
        frame_idx += 1
        frame = result.orig_img
        if frame is None:
            pbar.update(1); continue

        draw_hud(frame, args.sim_thr, args.seg_mode, args.infer)

        # no detections → dump frame if requested
        if result.boxes is None or len(result.boxes) == 0:
            if writer: writer.write(frame)
            pbar.update(1); continue

        ids = (result.boxes.id.cpu().numpy().astype(int)
               if hasattr(result.boxes, "id") and result.boxes.id is not None else None)
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        if ids is None or len(ids) == 0:
            ids = np.full((len(xyxy),), -1, dtype=int)

        # area sort → topk
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        order = np.argsort(-areas)[:args.topk]

        # optional one-time seg run (YOLO-seg)
        seg_result_for_frame = None
        if args.seg_mode == 'yolo' and seg_model is not None:
            try:
                seg_preds = list(seg_model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False))
                seg_result_for_frame = seg_preds[0] if len(seg_preds)>0 else None
            except Exception:
                seg_result_for_frame = None

        # collect crops & masks for batched CLIP
        crops, masks, meta = [], [], []  # meta keeps (i, tid, cx1,cy1,cx2,cy2)
        Hf, Wf = frame.shape[:2]
        for i in order:
            x1, y1, x2, y2 = xyxy[i]
            pad_x = args.context_pad * (x2 - x1)
            pad_y = args.context_pad * (y2 - y1)
            cx1 = max(0, int(x1 - pad_x)); cy1 = max(0, int(y1 - pad_y))
            cx2 = min(Wf - 1, int(x2 + pad_x)); cy2 = min(Hf - 1, int(y2 + pad_y))
            crop = frame[cy1:cy2, cx1:cx2]
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

            # save segmentation debug (independent)
            tid = int(ids[i])
            if args.save_mask and ((args.mask_for_all_dets) or (tid>=0)) and ((frame_idx % max(1, args.mask_stride))==0):
                ensure_dir(mask_root / f"{query_label}" / f"id_{tid}")
                mask_path = mask_root / f"{query_label}" / f"id_{tid}" / f"frame_{frame_idx:06d}_mask.png"
                cv2.imwrite(str(mask_path), roi_mask if roi_mask is not None else np.zeros((max(1, cy2-cy1), max(1, cx2-cx1)), np.uint8))
                if args.mask_overlay:
                    roi = frame[cy1:cy2, cx1:cx2].copy()
                    if roi_mask is not None:
                        color_mask = np.zeros_like(roi)
                        color_mask[roi_mask > 0] = (0, 255, 0)
                        cv2.addWeighted(color_mask, 0.35, roi, 0.65, 0, roi)
                    ov_path = mask_root / f"{query_label}" / f"id_{tid}" / f"frame_{frame_idx:06d}_overlay.jpg"
                    cv2.imwrite(str(ov_path), roi)

            if roi_mask is not None:
                masked = apply_mask_to_crop(crop, roi_mask, transparent=args.seg_transparent)
                ext = ".png" if args.seg_transparent else ".jpg"
                # (saving masked crops is currently optional/debug-only)

            if not args.seg_only:
                crops.append(crop)
                masks.append(roi_mask)
                meta.append((i, tid, (cx1, cy1, cx2, cy2)))

        # seg-only path (no CLIP/gating)
        if args.seg_only:
            for i in order:
                tid = int(ids[i])
                x1, y1, x2, y2 = xyxy[i]
                draw_box(frame, (x1, y1, x2, y2), f"{query_label} | id{tid} | conf {conf[i]:.2f} | SEG_ONLY", color=(0,255,255))
            if writer:
                writer.write(frame)
            pbar.update(1); continue

        # ---- batched CLIP on collected crops ----
        sims = torch.zeros(len(crops), dtype=torch.float32, device=embedder.device)
        crop_feats = None
        if len(crops) > 0:
            sims, crop_feats = embedder.sims_for_crops(
                crops_bgr=crops, masks=masks, use_mask=args.clip_use_mask,
                batch_size=args.clip_batch, exemplar_pack=exemplar_pack
            )
            sims = sims.detach().cpu()

        # optional debug dumps
        if debug_dir and args.dump_feats and crop_feats is not None:
            np.save(debug_dir / f"frame_{frame_idx:06d}_crop_feats.npy", crop_feats.detach().cpu().numpy())
            np.save(debug_dir / f"frame_{frame_idx:06d}_sims.npy", sims.numpy())

        # montages / histogram
        if debug_dir and args.viz_topk > 0 and len(crops) > 0:
            order_desc = torch.argsort(sims, descending=True).tolist()
            top_idx = order_desc[:min(args.viz_topk, len(crops))]
            bot_idx = order_desc[-min(args.viz_topk, len(crops)):]
            save_montage([crops[j] for j in top_idx], [float(sims[j]) for j in top_idx],
                         debug_dir / f"frame_{frame_idx:06d}_top.jpg")
            save_montage([crops[j] for j in bot_idx], [float(sims[j]) for j in bot_idx],
                         debug_dir / f"frame_{frame_idx:06d}_bot.jpg")
        if debug_dir and args.histogram and len(crops) > 0:
            hist = np.histogram(sims.numpy(), bins=20, range=(-0.2, 1.0))[0]
            Hh = 140; Ww = 320; canvas = np.ones((Hh, Ww, 3), np.uint8)*245
            mx = max(1, hist.max())
            for b, v in enumerate(hist):
                x1 = int(b * (Ww/20)); x2 = int((b+1) * (Ww/20) - 2)
                y2 = Hh - 10; y1 = y2 - int((v/mx)*(Hh-30))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (50,120,220), -1)
            cv2.putText(canvas, "sim histogram", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.imwrite(str(debug_dir / f"frame_{frame_idx:06d}_hist.png"), canvas)

        # render + persist
        for idx_batch, (i, tid, (cx1,cy1,cx2,cy2)) in enumerate(meta):
            x1, y1, x2, y2 = xyxy[i]
            sim = float(sims[idx_batch])
            keep = (sim >= args.sim_thr)
            # Only smooth when tid>=0 (i.e., tracking mode produced IDs)
            box_to_draw = (x1, y1, x2, y2) if tid < 0 else tuple(smoother.update(tid, (x1, y1, x2, y2)))

            # mask fg ratio for reasoning
            roi_mask = masks[idx_batch]
            fg_ratio = float((roi_mask>0).mean()) if roi_mask is not None and roi_mask.size>0 else 1.0
            reason = "OK" if keep else ("low_sim" if sim < args.sim_thr else "unknown")

            draw_box(frame, box_to_draw, f"{query_label} | id{tid} | conf {conf[i]:.2f} | sim {sim:.2f} [{reason}]",
                     color=(0,255,0) if keep else (0,0,255))

            if debug_dir:
                write_csv_row([frame_idx, tid, int(x1), int(y1), int(x2), int(y2),
                               int((x2-x1)*(y2-y1)), f"{fg_ratio:.3f}", f"{sim:.4f}", int(keep), reason])

            if keep and tid >= 0:
                tubes[int(tid)].append({
                    "t": float(frame_idx / (FPS if FPS>0 else 30.0)),
                    "f": int(frame_idx),
                    "box": [float(v) for v in np.asarray(box_to_draw, dtype=float).tolist()],
                    "cls": 0, "conf": float(conf[i]), "sim": sim
                })

        if writer: writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer: writer.release()
    if csv_writer: csv_writer.close()

    tubes_jsonable = {str(k): v for k, v in tubes.items()}
    with open(out_json, "w") as f:
        json.dump({"fps": float(FPS), "query_label": query_label, "tubes": tubes_jsonable}, f, indent=2)
    print("[DONE] Saved tubes:", out_json)
    if writer: print("[DONE] Saved video:", out_mp4)
    print("[DONE] Segmentation:", str(mask_root))
    print("[DONE] Crops:", str(objseg_root))
    if debug_dir: print("[DEBUG] CSV/NPY/montages:", str(debug_dir))

if __name__ == "__main__":
    main()
