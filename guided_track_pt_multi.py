#!/usr/bin/env python3
import os, json, argparse, random
from pathlib import Path
from collections import defaultdict
import cv2

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# ---------- helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_video_in(folder: Path) -> Path:
    vids = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    if not vids:
        vids = [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not vids:
        raise FileNotFoundError(f"No video found in {folder}")
    vids.sort(key=lambda p: p.stat().st_size, reverse=True)
    return vids[0]

def yolo_norm(x1, y1, x2, y2, W, H):
    # clamp to image bounds
    x1 = max(0, min(W - 1, float(x1))); x2 = max(0, min(W - 1, float(x2)))
    y1 = max(0, min(H - 1, float(y1))); y2 = max(0, min(H - 1, float(y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

def load_annotations(json_path: str, frame_base: int = 0):
    data = json.load(open(json_path, "r"))
    if isinstance(data, dict) and "videos" in data:
        items = data["videos"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Unknown JSON structure; expected list or dict with 'videos'")

    # Build: video_id -> frame_idx -> list of boxes
    per_video = {}
    for item in items:
        vid = str(item["video_id"])
        frames = defaultdict(list)
        for ann in item.get("annotations", []):
            for bb in ann.get("bboxes", []):
                f = int(bb["frame"]) - frame_base
                if f < 0:  # skip out-of-range if frame base was 1-based
                    continue
                x1 = float(bb["x1"]); y1 = float(bb["y1"])
                x2 = float(bb["x2"]); y2 = float(bb["y2"])
                frames[f].append((x1, y1, x2, y2))
        per_video[vid] = frames
    return per_video

def write_yaml(out_root: Path, use_lists: bool):
    if use_lists:
        yml = f"""# auto-generated
path: {out_root.resolve()}
train: {str((out_root / "train.txt").resolve())}
val: {str((out_root / "val.txt").resolve())}
nc: 1
names: [object]
"""
    else:
        yml = f"""# auto-generated
path: {out_root.resolve()}
train: images/train
val: images/val
nc: 1
names: [object]
"""
    (out_root / "data_objectness.yaml").write_text(yml)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Extract frames + one-class YOLO labels from annotations.json")
    ap.add_argument("--root", required=True, help="Dataset root with 'samples/' and 'annotations/annotations.json'")
    ap.add_argument("--out", default="yolo_frames", help="Output folder (under --root if relative)")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Split by video id")
    ap.add_argument("--stride", type=int, default=5, help="Save every Nth frame as negative/background (0=only positives)")
    ap.add_argument("--img_ext", default="jpg", choices=["jpg","png","jpeg","bmp"])
    ap.add_argument("--jpg_quality", type=int, default=90, help="JPEG quality when img_ext=jpg/jpeg")
    ap.add_argument("--frame_base", type=int, default=0, help="0 if annotations are 0-based, 1 if 1-based")
    ap.add_argument("--min_box", type=int, default=2, help="Drop boxes smaller than this (pixels) in width or height")
    ap.add_argument("--use_lists", action="store_true", help="Make YAML point to train.txt/val.txt instead of dirs")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    samples_dir = root / "samples"
    ann_path = root / "annotations" / "annotations.json"
    assert samples_dir.is_dir(), f"Missing {samples_dir}"
    assert ann_path.is_file(), f"Missing {ann_path}"

    # where to write
    out_root = (root / args.out) if not os.path.isabs(args.out) else Path(args.out)
    img_tr = out_root / "images" / "train"
    img_va = out_root / "images" / "val"
    lb_tr  = out_root / "labels" / "train"
    lb_va  = out_root / "labels" / "val"
    for p in [img_tr, img_va, lb_tr, lb_va]:
        ensure_dir(p)

    # parse JSON
    gt = load_annotations(str(ann_path), frame_base=args.frame_base)  # {vid: {frame_idx: [(x1,y1,x2,y2), ...]}}
    video_ids = sorted(gt.keys())
    random.shuffle(video_ids)

    # split by video
    n_val = max(1, int(len(video_ids) * args.val_ratio))
    val_ids = set(video_ids[:n_val])
    train_ids = set(video_ids[n_val:])

    train_list, val_list = [], []

    for vid in video_ids:
        vdir = samples_dir / vid
        try:
            vpath = find_video_in(vdir)
        except FileNotFoundError as e:
            print(f"[WARN] {e} — skipping {vid}")
            continue

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[WARN] Cannot open {vpath}, skipping {vid}")
            continue

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_w_boxes = set(gt[vid].keys())
        split = "val" if vid in val_ids else "train"
        img_dir = img_va if split == "val" else img_tr
        lb_dir  = lb_va if split == "val" else lb_tr

        fidx = -1
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1

            has_gt = (fidx in frames_w_boxes)
            save_bg = (args.stride > 0 and (fidx % args.stride == 0))
            if not (has_gt or save_bg):
                continue

            # image path
            img_name = f"{vid}_f{fidx:06d}.{args.img_ext}"
            img_path = img_dir / vid / img_name
            ensure_dir(img_path.parent)

            # write image
            if args.img_ext.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality])
            else:
                cv2.imwrite(str(img_path), frame)

            # write label (class 0; multiple boxes per frame ok)
            lbl_path = lb_dir / vid / (img_name.rsplit(".", 1)[0] + ".txt")
            ensure_dir(lbl_path.parent)
            lines = []
            for (x1, y1, x2, y2) in gt[vid].get(fidx, []):
                if (x2 - x1) < args.min_box or (y2 - y1) < args.min_box:
                    continue
                xc, yc, w, h = yolo_norm(x1, y1, x2, y2, W, H)
                # clamp to [0,1] after norm (numerical safety)
                xc = max(0.0, min(1.0, xc)); yc = max(0.0, min(1.0, yc))
                w  = max(0.0, min(1.0,  w)); h  = max(0.0, min(1.0,  h))
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            lbl_path.write_text("\n".join(lines))

            # record list (absolute path so YAML list works anywhere)
            if split == "val":
                val_list.append(str(img_path.resolve()))
            else:
                train_list.append(str(img_path.resolve()))
            saved += 1

        cap.release()
        print(f"[{vid}] saved {saved} frames -> split:{split}")

    # write yaml + lists
    (out_root / "train.txt").write_text("\n".join(train_list))
    (out_root / "val.txt").write_text("\n".join(val_list))
    write_yaml(out_root, use_lists=args.use_lists)

    print("\nDone.")
    print("YOLO YAML:", out_root / "data_objectness.yaml")
    print("Train list:", out_root / "train.txt")
    print("Val   list:", out_root / "val.txt")
    print("Tip: set --stride 0 to save ONLY positive frames; increase it to add background negatives.")

if __name__ == "__main__":
    main()
