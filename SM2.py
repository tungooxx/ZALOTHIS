#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
import torchvision as tv
from tqdm import tqdm
import numpy as np
# Data utilities
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.rpn import RPNHead

try:
    from Data2 import make_loaders, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    raise SystemExit("Place Data2.py in the same folder or in PYTHONPATH.")


# --------------------- SIMILARITY HEAD ---------------------
class SupportConditionedRPNHead(RPNHead):

    def __init__(self, in_channels: int, num_anchors: int, proto_dim: int):
        super().__init__(in_channels, num_anchors)
        # Project prototype dim (e.g. 1024) -> RPN feature dim (e.g. 256)
        self.proto_proj = nn.Linear(proto_dim, in_channels)
        # Learnable scale for how strongly prototype influences the features
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # Will store (D,) or (1, D) prototype on each episode
        self.prototype: torch.Tensor | None = None

    @torch.no_grad()
    def set_prototype(self, proto: torch.Tensor):
        """
        proto: (K, D) or (1, D) or (D,)
        We aggregate to a single mean prototype for RPN conditioning.
        """
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)  # (1, D)
        # (K, D) -> (D,)
        proto_mean = proto.mean(dim=0)
        self.prototype = F.normalize(proto_mean, dim=-1)

    def clear_prototype(self):
        self.prototype = None

    def forward(self, x):
        """
        x: list[Tensor], each of shape (B, C, H, W). Standard RPNHead API.
        Returns:
          objectness: list[Tensor]
          pred_bbox_deltas: list[Tensor]
        """
        objectness = []
        pred_bbox_deltas = []

        # Precompute prototype feature if we have one
        proto_feat = None
        if self.prototype is not None:
            # (D,) -> (1, C, 1, 1)
            proto_feat_vec = self.proto_proj(self.prototype)  # (C,)
            proto_feat = proto_feat_vec.view(1, -1, 1, 1)

        for feature in x:
            t = feature
            if proto_feat is not None:
                # Broadcast to current feature device/shape
                t = t + self.alpha * proto_feat.to(feature.device)

            t = F.relu(self.conv(t))
            objectness.append(self.cls_logits(t))
            pred_bbox_deltas.append(self.bbox_pred(t))

        return objectness, pred_bbox_deltas

class CrossAttentionSimilarityHead(nn.Module):

    def __init__(self, in_channels, num_classes=2, nhead=8):
        super().__init__()
        self.d_model = in_channels       # e.g. 1024
        self.nhead = nhead

        # Cross-attention: Q from RoIs, K/V from prototypes
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            dropout=0.1,
            batch_first=False,          # (seq, batch, dim)
        )

        # Optional small feed-forward to post-process attended features
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        # Similarity-based classifier: one scalar per RoI
        self.cls_score = nn.Linear(self.d_model, 1)

        # Class-specific box regression (2 classes: bg, fg)
        self.bbox_pred = nn.Linear(self.d_model, num_classes * 4)

        # Will be set to shape (K, D) per episode
        self.prototype = None

    def set_prototype(self, proto: torch.Tensor):
        """
        proto: (K, D) or (D,) tensor.
        - If (D,), treat as a single prototype (K=1).
        - If (K, D), keep them all (per-support).
        """
        if proto.dim() == 1:
            proto = proto.unsqueeze(0)  # (1, D)
        self.prototype = F.normalize(proto, dim=-1)  # (K, D)

    def forward(self, x: torch.Tensor):
        """
        x: (N, D) RoI features from box_head.

        Returns:
          scores:      (N, 2)          logits for [background, foreground]
          bbox_deltas: (N, 2 * 4)
        """
        if self.prototype is None:
            raise ValueError("Prototype not set in CrossAttentionSimilarityHead. "
                             "Call set_prototype() before forward().")

        # Normalize RoI features
        x = F.normalize(x, dim=-1)             # (N, D)
        proto = F.normalize(self.prototype, dim=-1)  # (K, D)

        N = x.shape[0]
        K = proto.shape[0]

        # MultiheadAttention expects (seq_len, batch, embed_dim)
        # We'll use batch_size = 1 and treat all RoIs as a sequence.
        q = x.unsqueeze(1)             # (N, 1, D)
        k = proto.unsqueeze(1)         # (K, 1, D)
        v = k                          # (K, 1, D)

        # Cross-attention: RoIs attend to prototypes
        att_out, _ = self.cross_attn(q, k, v)   # (N, 1, D)
        att_out = att_out.squeeze(1)            # (N, D)

        # Residual + simple FFN (like a tiny transformer block)
        attended_x = x + self.ffn(att_out)      # (N, D)

        # --- Similarity-based classification ---
        sim = self.cls_score(attended_x).squeeze(-1)   # (N,)
        scores = torch.stack([-sim, sim], dim=1)       # (N, 2)

        # --- Box regression ---
        bbox_deltas = self.bbox_pred(attended_x)       # (N, 2 * 4)

        return scores, bbox_deltas




# --------------------- MODEL BUILDER ---------------------
# def build_siamese_model(agnostic_weights=None):
#     model = tv.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
#     roi_feat_dim = model.roi_heads.box_head.fc7.out_features
#     model.roi_heads.box_predictor = SimilarityHead(
#         in_channels=roi_feat_dim,
#         proto_channels=roi_feat_dim,
#         num_classes=2
#     )
#     if agnostic_weights:
#         ckpt = torch.load(agnostic_weights, map_location="cpu")["model"]
#         keys_to_remove = [k for k in ckpt.keys() if k.startswith("roi_heads.box_predictor.")]
#         for k in keys_to_remove:
#             del ckpt[k]
#         missing, unexpected = model.load_state_dict(ckpt, strict=False)
#         print(f"Loaded weights. Missing: {missing}, Unexpected: {unexpected}")
#     return model


def build_siamese_model(agnostic_weights=None):
    model = tv.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    # RoI feature dimension from the box head (e.g. 1024)
    roi_feat_dim = model.roi_heads.box_head.fc7.out_features

    # ---------------- RPN: replace head with SupportConditionedRPNHead ----------------
    # In torchvision, FasterRCNN builds the RPN head from:
    #   in_channels = backbone.out_channels
    #   num_anchors = anchor_generator.num_anchors_per_location()[0]
    rpn_in_channels = model.backbone.out_channels
    num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]

    model.rpn.head = SupportConditionedRPNHead(
        in_channels=rpn_in_channels,
        num_anchors=num_anchors,
        proto_dim=roi_feat_dim,   # prototypes come from RoI head
    )

    # ---------------- RoI head: Cross-attention similarity head ----------------
    model.roi_heads.box_predictor = CrossAttentionSimilarityHead(
        in_channels=roi_feat_dim,
        num_classes=2,
        nhead=8,
    )

    if agnostic_weights:
        print(f"Loading weights from: {agnostic_weights}")
        ckpt = torch.load(agnostic_weights, map_location="cpu")["model"]

        # Remove old predictor weights (RPN + RoI box head) so we can re-train them
        print("Filtering checkpoint keys...")
        keys_to_remove = [
            k for k in ckpt.keys()
            if k.startswith("roi_heads.box_predictor.")
            or k.startswith("rpn.head.")
        ]
        for k in keys_to_remove:
            print(f"  - Removing pre-trained key: {k}")
            del ckpt[k]

        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"Weights loaded. Missing keys (new episodic heads): {missing}")
        print(f"Unexpected keys (should be empty): {unexpected}")

    return model


# --------------------- SUPPORT PROTOTYPE ---------------------
@torch.no_grad()
def make_support_proto(model, support_tensors, device, aggregate=True):
    """
    Build prototypes from support images.

    support_tensors: (K, C, H, W) already normalized.

    Returns:
      proto_all: (K, D) if aggregate=False
                 (1, D) if aggregate=True  (mean prototype over K)
    """
    model.eval()

    supp = support_tensors.to(device).float()
    K = supp.shape[0]
    feats_all = []

    for i in range(K):
        img = supp[i:i + 1]  # (1, C, H, W)
        feats = model.backbone(img)
        _, _, H, W = img.shape

        # full-image RoI, but we've already center-cropped, so it is mostly object
        boxes = [torch.tensor([[0., 0., W - 1, H - 1]], device=device)]
        shapes = [(H, W)]

        pooled = model.roi_heads.box_roi_pool(feats, boxes, shapes)  # (1, D, ph, pw)
        feat = model.roi_heads.box_head(pooled)                      # (1, D)
        feat = feat.squeeze(0)                                      # (D,)
        feats_all.append(feat)

    proto_all = torch.stack(feats_all, dim=0)  # (K, D)

    if aggregate:
        proto_all = proto_all.mean(dim=0, keepdim=True)  # (1, D)

    proto_all = F.normalize(proto_all, dim=-1)
    return proto_all




# --------------------- DENORMALIZE ---------------------
def denormalize_for_model(img_tensor, device):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


# --------------------- TRAIN ONE EPISODE ---------------------
def train_one_episode(model, episode_batch, optimizer, device, scaler=None, max_norm=1.0):
    supports, query_img, target = episode_batch
    supports = supports[0].to(device)
    query_img = query_img[0].to(device)
    tgt = {
        "boxes": target["boxes"][0].to(device),
        "labels": target["labels"][0].to(device)
    }

    # --- 1. Create prototype ---
    with torch.no_grad():
        proto = make_support_proto(model, supports, device)

    # Set prototype for both RPN and RoI head
    model.roi_heads.box_predictor.set_prototype(proto)
    if hasattr(model.rpn.head, "set_prototype"):
        model.rpn.head.set_prototype(proto)

    # --- 2. Forward + Loss ---
    query_img_0_1 = denormalize_for_model(query_img, device)

    model.train()
    loss_dict = model([query_img_0_1], [tgt])
    losses = sum(loss for loss in loss_dict.values())

    # Compute episodic recall
    model.eval()  # switch to eval
    ep_recall = 0.0  # Default value
    with torch.no_grad():
        preds = model([query_img_0_1])[0]
        keep = (preds["labels"] == 1) & (preds["scores"] >= 0.5)
        boxes_pred = preds["boxes"][keep]
        boxes_gt = tgt["boxes"]
        if boxes_gt.numel() == 0 or boxes_pred.numel() == 0:
            ep_recall = 0.0
        else:
            ious = box_iou(boxes_pred, boxes_gt)
            ep_recall = (ious.max(dim=0).values >= 0.5).float().mean().item()

    model.train()  # switch back to training for next step

    # --- 3. Backward ---
    optimizer.zero_grad(set_to_none=True)
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

    # Clear episodic prototypes (RPN + RoI head)
    model.roi_heads.box_predictor.prototype = None
    if hasattr(model.rpn.head, "clear_prototype"):
        model.rpn.head.clear_prototype()

    # --- 4. Episodic recall (optional) ---
    ### THIS BLOCK WAS THE BUG AND HAS BEEN REMOVED.
    ### The model is in train() mode here, but this block
    ### called model() without targets, causing the assertion error.
    ### The `ep_recall` from the first block is all we need.

    # Convert loss_dict to floats for logging
    loss_dict_detached = {k: float(v.detach().item()) for k, v in loss_dict.items()}
    loss_dict_detached["loss"] = float(losses.detach().item())
    return loss_dict_detached, ep_recall


# --------------------- EVALUATE ---------------------
@torch.no_grad()
def evaluate_siamese(model, val_loader, device, iou_thresh=0.5):
    """
    Single-class evaluation:
      - Computes precision/recall curve and AP (area under PR curve)
      - Also returns recall@IoU and precision@IoU using a 0.5 score threshold.
    """
    model.eval()

    all_scores = []
    all_tp_flags = []
    total_gt = 0

    # For thresholded precision/recall
    total_hit_thr = 0
    total_pred_thr = 0

    for supports, query_img, target in tqdm(val_loader, desc="Validating"):
        supports = supports[0].to(device)
        query_img = query_img[0].to(device)
        boxes_gt = target["boxes"][0].to(device)
        labels_gt = target["labels"][0].to(device)

        # Only count positives as GTs (labels==1)
        pos_mask = labels_gt == 1
        boxes_gt = boxes_gt[pos_mask]
        num_gt = boxes_gt.shape[0]
        total_gt += num_gt

        proto = make_support_proto(model, supports, device)
        model.roi_heads.box_predictor.set_prototype(proto)
        if hasattr(model.rpn.head, "set_prototype"):
            model.rpn.head.set_prototype(proto)

        query_img_0_1 = denormalize_for_model(query_img, device)
        preds = model([query_img_0_1])[0]

        # Clear prototypes
        model.roi_heads.box_predictor.prototype = None
        if hasattr(model.rpn.head, "clear_prototype"):
            model.rpn.head.clear_prototype()

        if num_gt == 0 and len(preds["boxes"]) == 0:
            continue

        # Only foreground predictions
        fg_mask = preds["labels"] == 1
        boxes_pred = preds["boxes"][fg_mask].to(device)
        scores_pred = preds["scores"][fg_mask].to(device)

        # --------- For PR curve / AP (use ALL scores) ----------
        if num_gt > 0 and boxes_pred.numel() > 0:
            ious = box_iou(boxes_pred, boxes_gt)  # (num_pred, num_gt)

            # For each prediction, find best GT
            best_ious, best_gt_idx = ious.max(dim=1)

            # Keep track of which GTs have been matched already
            gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=device)

            for s, iou, g_idx in zip(scores_pred, best_ious, best_gt_idx):
                if iou >= iou_thresh and not gt_matched[g_idx]:
                    all_scores.append(float(s))
                    all_tp_flags.append(1)
                    gt_matched[g_idx] = True
                else:
                    all_scores.append(float(s))
                    all_tp_flags.append(0)
        else:
            # No GTs or no predictions: all predictions are false positives
            for s in scores_pred:
                all_scores.append(float(s))
                all_tp_flags.append(0)

        # --------- For simple precision/recall using score >= 0.5 ----------
        keep_thr = scores_pred >= 0.5
        boxes_pred_thr = boxes_pred[keep_thr]

        total_pred_thr += boxes_pred_thr.shape[0]
        if num_gt > 0 and boxes_pred_thr.numel() > 0:
            ious_thr = box_iou(boxes_pred_thr, boxes_gt)
            hits_thr = (ious_thr.max(dim=0).values >= iou_thresh).sum().item()
            total_hit_thr += hits_thr

    # ======= Compute precision/recall curve & AP =======
    if len(all_scores) == 0:
        return {
            "ap": 0.0,
            "pr_curve": ([], []),
            "recall": 0.0,
            "precision": 0.0,
        }

    scores_np = np.array(all_scores)
    tps_np = np.array(all_tp_flags, dtype=np.int32)

    # Sort by descending score
    order = np.argsort(-scores_np)
    tps_sorted = tps_np[order]

    fps_sorted = 1 - tps_sorted

    cum_tps = np.cumsum(tps_sorted)
    cum_fps = np.cumsum(fps_sorted)

    recalls = cum_tps / max(total_gt, 1)
    precisions = cum_tps / np.maximum(cum_tps + cum_fps, 1e-8)

    # AP as area under PR curve (trapezoidal)
    # First, enforce monotonically non-increasing precision (classic VOC-style)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Integrate
    ap = 0.0
    for i in range(1, mrec.size):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    # Simple precision/recall at threshold
    recall_thr = total_hit_thr / max(total_gt, 1)
    precision_thr = total_hit_thr / max(total_pred_thr, 1)

    return {
        "ap": float(ap),
        "pr_curve": (recalls.tolist(), precisions.tolist()),
        "recall": float(recall_thr),
        "precision": float(precision_thr),
    }




# --------------------- BUILD OPTIMIZER ---------------------
def build_optimizer(model, lr_head, backbone_lr_scale, freeze_backbone=False):
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.roi_heads.box_predictor.parameters())
    # Make sure we only add params that require gradients
    param_groups = [{"params": [p for p in head_params if p.requires_grad], "lr": lr_head}]

    if not freeze_backbone:
        param_groups.append(
            {"params": [p for p in backbone_params if p.requires_grad], "lr": lr_head * backbone_lr_scale})
    else:
        print("Freezing backbone parameters.")
        for p in backbone_params:
            p.requires_grad = False

    # Filter out empty param groups
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]

    if not param_groups:
        raise ValueError("No parameters to optimize. Check if model is frozen.")

    optimizer = optim.AdamW(param_groups, lr=lr_head, weight_decay=1e-4)
    return optimizer


# --------------------- MAIN ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--agnostic_weights", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1)  # Episodic, so batch_size=1 is expected
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--backbone_lr_scale", type=float, default=0.1)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--unfreeze_at_epoch", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="checkpoints_siamese")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from.")

    args = ap.parse_args()

    # --- Setup Loaders ---
    loaders = make_loaders(
        data_root=Path(args.data_root),
        annot_path=Path(args.data_root) / "annotations/annotations.json",
        samples_root=Path(args.data_root) / "samples",
        batch_size_train=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader = loaders["train_episode_loader"]
    val_loader = loaders["val_episode_loader"]

    device = torch.device(args.device)

    # --- Setup Model ---
    model = build_siamese_model(args.agnostic_weights).to(device)
    scaler = torch.amp.GradScaler()  # Not used, but here

    start_epoch = 1
    best_recall = -1.0
    best_path = None

    if args.resume_from:
        print(f"Resuming training from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)

        # Load model weights
        model.load_state_dict(ckpt["model"])

        # Set epoch and best recall
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("val_recall", -1.0)  # .get() for backward compatibility
        best_path = args.resume_from
        print(f"Resuming from start of Epoch {start_epoch}. Best recall so far: {best_recall:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    current_freeze_state = args.freeze_backbone
    if args.freeze_backbone and start_epoch > args.unfreeze_at_epoch:
        print(f"Resuming after unfreeze epoch ({args.unfreeze_at_epoch}). Starting with backbone unfrozen.")
        current_freeze_state = False  # We've already passed the unfreeze epoch

    optimizer = build_optimizer(model, args.lr_head, args.backbone_lr_scale, current_freeze_state)

    print("Starting improved Siamese episodic training...")
    if args.resume_from:
        print(f"Optimizer re-built with new LR: lr_head={args.lr_head}, backbone_lr_scale={args.backbone_lr_scale}")

    # --- Loop starts from `start_epoch` ---
    for epoch in range(start_epoch, args.epochs + 1):
        # linearly ramp from 0.1 to 0.6, for example
        neg_prob = min(0.8, 0.1 + 0.05 * (epoch-10 - 1))
        train_loader.dataset.negative_prob = neg_prob
        print(f"Epoch {epoch}: negative_prob = {neg_prob:.2f}")
        # Check if we need to unfreeze *this* epoch
        if current_freeze_state and epoch == args.unfreeze_at_epoch:
            print(f"Unfreezing backbone at epoch {epoch}")
            # Rebuild optimizer to include backbone params
            optimizer = build_optimizer(model, args.lr_head, args.backbone_lr_scale, freeze_backbone=False)
            current_freeze_state = False  # Mark as unfrozen

        model.train()
        running_loss = 0.0
        running_recall = 0.0

        for i, episode_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} train")):
            loss_dict, ep_recall = train_one_episode(model, episode_batch, optimizer, device, scaler)
            running_loss += loss_dict["loss"]
            running_recall += ep_recall

            if (i + 1) % 50 == 0:
                print(
                    f"  Iter {i + 1}/{len(train_loader)} | avg_loss={running_loss / (i + 1):.4f} | avg_recall={running_recall / (i + 1):.4f}")

        avg_loss = running_loss / len(train_loader)
        avg_recall = running_recall / len(train_loader)
        print(f"[Epoch {epoch}] Train done. Avg_loss={avg_loss:.4f}, Avg_recall={avg_recall:.4f}")

        metrics = evaluate_siamese(model, val_loader, device)
        print(
            f"[Epoch {epoch}] "
            f"Val AP={metrics['ap']:.4f}, "
            f"Recall@0.5={metrics['recall']:.4f}, "
            f"Precision@0.5={metrics['precision']:.4f}"
        )
        val_recall = metrics["recall"]  # if you still want to track 'best_recall'

        ckpt_path = os.path.join(args.out_dir, f"siamese_mbv3_epoch{epoch:02d}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_recall": val_recall}, ckpt_path)

        if val_recall > best_recall:
            best_recall = val_recall
            best_path = ckpt_path
            print(f"  -> New best recall saved at {best_path}")

    print(f"\nTraining done. Best recall={best_recall:.4f} @ {best_path}")


if __name__ == "__main__":
    main()