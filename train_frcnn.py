#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from Data import make_loaders, collate_fn_detection, IMAGENET_MEAN, IMAGENET_STD
except ImportError as e:
    raise SystemExit("Please place data_setup.py (from the previous message) in the same folder or in PYTHONPATH.")


def maybe_denormalize_images(images, device=None, dtype=None):
    """
    Undo dataset ImageNet normalization back to [0,1] so torchvision's
    GeneralizedRCNNTransform can apply its own normalization once.
    """
    if device is None:
        device = images[0].device
    if dtype is None:
        dtype = images[0].dtype

    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device, dtype=dtype).view(3, 1, 1)

    out = []
    for img in images:
        if img.dim() != 3 or img.size(0) != 3:
            raise ValueError("Expected CxHxW tensor with 3 channels")
        out.append((img * std + mean).clamp(0.0, 1.0))
    return out


def build_model(num_classes=2, weights="DEFAULT"):
    # Faster R-CNN with MobileNetV3 Large FPN backbone
    model = tv.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = tv.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device, epoch, print_freq=50, max_norm=1.0):
    model.train()
    running = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    t0 = time.time()

    for it, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        images = maybe_denormalize_images(images, device=device)

        tgts = []
        for t in targets:
            tgts.append({
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "image_id": t["image_id"].to(device),
            })

        loss_dict = model(images, tgts)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # --- logging (fix) ---
        loss_total = float(losses.detach().item())
        # detach individual losses for logging
        loss_dict_detached = {k: float(v.detach().item()) for k, v in loss_dict.items()}

        # accumulate
        running.setdefault("loss", 0.0)
        running["loss"] += loss_total
        for k, v in loss_dict_detached.items():
            running[k] = running.get(k, 0.0) + v

        if (it + 1) % print_freq == 0:
            seen = it + 1
            msg = f"[epoch {epoch:02d} iter {seen:05d}/{len(loader):05d}] "
            msg += " ".join([f"{k}={running[k] / seen:.4f}" for k in sorted(running.keys())])
            print(msg)

    dt = time.time() - t0
    seen = len(loader)
    epoch_stats = {k: running[k] / max(1, seen) for k in running}
    print(f"[epoch {epoch:02d}] done in {dt:.1f}s | " + " ".join([f"{k}={v:.4f}" for k, v in epoch_stats.items()]))
    return epoch_stats


@torch.no_grad()
def quick_val_recall(model, val_episode_loader, device, score_thresh=0.5, iou_thresh=0.5, max_dets=200):
    """
    Quick sanity-check metric: Recall@IoU>=0.5 on base validation episodes
    (class-agnostic): fraction of GT boxes that are matched by any predicted box.
    """
    from torchvision.ops import box_iou

    model.eval()
    total_gt = 0
    total_hit = 0

    for supports, query_img, target in val_episode_loader:
        # supports are unused in detection-only validation, we just ensure recall is reasonable
        img = query_img.to(device)[0]

        # denormalize since dataset applied normalization
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(3,1,1)
        std = torch.tensor(IMAGENET_STD, device=device).view(3,1,1)
        img = (img * std + mean).clamp(0,1)

        preds = model([img])[0]
        keep = preds["scores"] >= score_thresh
        boxes_pred = preds["boxes"][keep][:max_dets]

        boxes_gt = target["boxes"][0].to(device)
        total_gt += boxes_gt.shape[0]

        if boxes_gt.numel() == 0:
            continue
        if boxes_pred.numel() == 0:
            continue

        ious = box_iou(boxes_pred, boxes_gt)  # [Np, Ng]
        # A GT is recalled if any pred IoU >= thresh
        hits = (ious.max(dim=0).values >= iou_thresh).sum().item()
        total_hit += hits

    recall = (total_hit / total_gt) if total_gt > 0 else 0.0
    return recall


# ---------- small helper to infer num_classes ----------
def infer_num_classes_from_loader(loader, max_batches=50):
    """Scan a few batches from loader to find max label index -> infer num_classes"""
    max_label = -1
    scanned = 0
    for images, targets in loader:
        for t in targets:
            if t["labels"].numel() == 0:
                continue
            lbl = int(t["labels"].max().item())
            if lbl > max_label:
                max_label = lbl
        scanned += 1
        if scanned >= max_batches:
            break
    return int(max_label + 1) if max_label >= 0 else 1


# ---------- function to load resume checkpoint ----------
def load_checkpoint_if_any(args, model, optimizer=None, map_location=None):
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=map_location)
        # ckpt could store {"model": state_dict, ...} or raw state_dict
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            print("Loaded model.state_dict from checkpoint (strict=False).")
            if optimizer is not None and "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    print("Loaded optimizer state.")
                except Exception as e:
                    print(f"Could not load optimizer state: {e}")
        elif isinstance(ckpt, dict):
            # maybe it's a state dict directly
            try:
                model.load_state_dict(ckpt, strict=False)
                print("Loaded model.state_dict from checkpoint.")
            except Exception as e:
                print(f"Failed to load checkpoint keys: {e}")
        else:
            print("Unrecognized checkpoint format.")
    else:
        if args.resume:
            print(f"Resume path specified but not found: {args.resume}")


# ----------------- Albumentations transforms & wrapper -----------------
def get_train_transforms():
    return A.Compose([
        # random rescale / multi-scale effect: here we resize longest side to 800 then random crop
        A.Resize(height=800, width=800),

        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Blur(blur_limit=3, p=0.1),

        # Keep final output as tensor (C,H,W) in float [0,1]
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))


def get_val_transforms():
    return A.Compose([
        A.Resize(height=800, width=800),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


class AugmentedDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform, imagenet_mean, imagenet_std):
        self.base = base_dataset
        self.transform = transform
        self.mean = torch.tensor(imagenet_mean).view(3,1,1)
        self.std = torch.tensor(imagenet_std).view(3,1,1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]  # original dataset output
        # Expect img: Tensor C,H,W normalized. Denormalize to 0..1 H,W,C numpy
        if isinstance(img, torch.Tensor):
            img_cpu = img.detach().cpu()
            img_denorm = (img_cpu * self.std + self.mean).permute(1,2,0).numpy()  # H,W,C in 0..1
        else:
            # already numpy HWC 0..1 or 0..255 - try to normalize to 0..1
            import numpy as _np
            img_np = img
            if img_np.max() > 2:
                img_denorm = (img_np.astype('float32') / 255.0)
            else:
                img_denorm = img_np

        # Prepare bboxes in pascal_voc format (x_min,y_min,x_max,y_max) absolute pixels
        boxes = target['boxes'].detach().cpu().numpy().tolist() if isinstance(target['boxes'], torch.Tensor) else target['boxes']
        labels = target['labels'].detach().cpu().numpy().tolist() if isinstance(target['labels'], torch.Tensor) else target['labels']

        # Albumentations expects bboxes as list of lists and labels list
        transformed = self.transform(image=img_denorm, bboxes=boxes, labels=labels)
        img_t = transformed['image']  # ToTensorV2 -> tensor C,H,W in float [0,1]

        new_boxes = transformed['bboxes']
        new_labels = transformed['labels']

        # convert to tensors in required format
        if len(new_boxes) == 0:
            # Ensure at least zero-sized boxes handled (empty tensor)
            boxes_tensor = torch.zeros((0,4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(new_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(new_labels, dtype=torch.int64)

        # image_id keep from original target if present; else use idx
        image_id = target.get('image_id', torch.tensor([idx]))
        if isinstance(image_id, torch.Tensor) and image_id.numel() == 1:
            image_id = image_id
        else:
            image_id = torch.tensor([int(idx)])

        new_target = {'boxes': boxes_tensor, 'labels': labels_tensor, 'image_id': image_id}
        return img_t, new_target


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=".", help="root containing annotations/ and samples/")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)  # fix name
    ap.add_argument("--novel_labels", type=str, default="", help="comma-separated class names to treat as novel")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to load weights from")
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone (useful as warmup to avoid overfitting).")
    ap.add_argument("--score_thresh", type=float, default=0.2, help="score threshold used in quick_val_recall for debug")
    ap.add_argument("--use_augment", action="store_true", help="Enable training-time augmentations (albumentations).")
    args = ap.parse_args()

    # Build loaders (uses your file tree & annotations)
    novel = [s for s in args.novel_labels.split(",") if s] if args.novel_labels else None
    loaders = make_loaders(
        data_root=Path(args.data_root),
        annot_path=Path(args.data_root) / "annotations" / "annotations.json",
        samples_root=Path(args.data_root) / "samples",
        novel_labels=novel,
        batch_size_train=args.batch_size,
        num_workers=args.num_workers,
    )

    train_loader = loaders["train_loader"]
    val_episode_loader = loaders["val_episode_loader"]

    # Optionally wrap train dataset with augmentations
    if args.use_augment:
        print("Applying albumentations-based augmentations to training dataset.")
        train_dataset_base = train_loader.dataset
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()  # kept for reference if you want to wrap val too

        wrapped_train_dataset = AugmentedDatasetWrapper(train_dataset_base, train_transform, IMAGENET_MEAN, IMAGENET_STD)

        # recreate a DataLoader with same options but wrapped dataset
        train_loader = DataLoader(
            wrapped_train_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn_detection,
            pin_memory=True
        )

    # infer num_classes from loader (safer than hardcode)
    num_classes = infer_num_classes_from_loader(train_loader)
    print(f"Inferred num_classes={num_classes} from train loader (check if correct).")

    device = torch.device(args.device)
    model = build_model(num_classes=num_classes, weights="DEFAULT").to(device)

    if args.freeze_backbone:
        # freeze backbone parameters (common warm-up trick)
        for name, p in model.backbone.named_parameters():
            p.requires_grad = False
        print("Backbone parameters frozen for warm-up.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.66*args.epochs), int(0.88*args.epochs)], gamma=0.1)

    # load resume if any
    load_checkpoint_if_any(args, model, optimizer=optimizer, map_location=device)

    os.makedirs(args.out_dir, exist_ok=True)
    best_recall = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, print_freq=50, max_norm=1.0)

        # quick recall sanity check (use args.score_thresh)
        recall = quick_val_recall(model, val_episode_loader, device, score_thresh=args.score_thresh, iou_thresh=0.5)
        print(f"[epoch {epoch:02d}] quick base-val Recall@0.5 (score>={args.score_thresh}) = {recall:.4f}")

        # Save checkpoint (also save optimizer + lr scheduler states)
        ckpt_path = os.path.join(args.out_dir, f"frcnn_mbv3_epoch{epoch:02d}.pth")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "recall": recall
        }, ckpt_path)
        if recall > best_recall:
            best_recall = recall
            best_path = ckpt_path

        lr_scheduler.step()

    print(f"Training done. Best recall={best_recall:.4f} @ {best_path}")
    print("Tip: next run eval_with_support.py to apply the support-prototype cosine filter on episodes.")


if __name__ == "__main__":
    main()
