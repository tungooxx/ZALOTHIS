#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import box_iou
import torchvision as tv
from tqdm import tqdm
from torchvision.models.detection.rpn import RPNHead

# --------------------- DATA UTILS ---------------------
try:
    from Data2 import make_loaders, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    raise SystemExit("Place Data2.py in the same folder or in PYTHONPATH.")


# =====================================================
#  1) PROTOTYPE ENCODER
# =====================================================

class PrototypeEncoder(nn.Module):
    """
    Next-gen prototype encoder.

    Input:
      - support_feats: (K, D) RoI features from support images

    Operations:
      - LayerNorm
      - Multi-head self-attention over the K support tokens
      - MLP + residual
      - Outputs:
          proto_tokens: (K, D) refined support tokens
          proto_global: (D,)   global prototype (mean pooled)
    """

    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.d_model = d_model

        self.ln_in = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True,  # (B, K, D)
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, support_feats: torch.Tensor):
        """
        support_feats: (K, D)
        returns:
          proto_tokens: (K, D)
          proto_global: (D,)
        """
        if support_feats.dim() != 2:
            raise ValueError(f"support_feats must be (K,D), got {support_feats.shape}")
        K, D = support_feats.shape
        x = self.ln_in(support_feats)            # (K, D)
        x = x.unsqueeze(0)                       # (1, K, D) as a batch of 1

        att_out, _ = self.self_attn(x, x, x)     # (1, K, D)
        x = x + att_out                          # residual
        x = x + self.ffn(x)                      # residual + FFN
        x = self.ln_out(x)                       # (1, K, D)

        proto_tokens = x.squeeze(0)              # (K, D)
        proto_tokens = F.normalize(proto_tokens, dim=-1)

        proto_global = proto_tokens.mean(dim=0)  # (D,)
        proto_global = F.normalize(proto_global, dim=-1)

        return proto_tokens, proto_global


# =====================================================
#  2) SUPPORT-CONDITIONED RPN
# =====================================================

class SupportConditionedRPNHead(RPNHead):
    """
    RPN head conditioned on the *global* prototype.

    - Standard RPNHead cnn + cls_logits + bbox_pred
    - We inject a projected prototype bias into features at each level:

        F'_l = F_l + alpha * proj(proto_global)

    This biases proposals toward regions consistent with the current support class.
    """

    def __init__(self, in_channels: int, num_anchors: int, proto_dim: int):
        super().__init__(in_channels, num_anchors)
        self.proto_proj = nn.Linear(proto_dim, in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # start slightly conservative
        self.prototype_global: torch.Tensor | None = None

    @torch.no_grad()
    def set_prototype_global(self, proto_global: torch.Tensor):
        """
        proto_global: (D,) or (1,D)
        """
        if proto_global.dim() == 2:
            proto_global = proto_global.squeeze(0)
        self.prototype_global = F.normalize(proto_global, dim=-1)

    def clear_prototype(self):
        self.prototype_global = None

    def forward(self, x):
        """
        x: list[Tensor], each (B, C, H, W)
        """
        objectness = []
        pred_bbox_deltas = []

        proto_feat = None
        if self.prototype_global is not None:
            v = self.proto_proj(self.prototype_global)   # (C,)
            proto_feat = v.view(1, -1, 1, 1)             # (1,C,1,1)

        for feature in x:
            t = feature
            if proto_feat is not None:
                t = t + self.alpha * proto_feat.to(feature.device)
            t = F.relu(self.conv(t))
            objectness.append(self.cls_logits(t))
            pred_bbox_deltas.append(self.bbox_pred(t))

        return objectness, pred_bbox_deltas


# =====================================================
#  3) CROSS-ATTENTION ROI HEAD
# =====================================================

class CrossAttentionSimilarityHead(nn.Module):
    """
    Cross-attention-based box head:

      - RoI features:   x ∈ R^{N x D}  (queries)
      - Prototypes:     P ∈ R^{K x D}  (keys & values)

    Multi-head cross-attention produces support-aware RoI embeddings,
    used by:
      - similarity classifier -> logits for [bg, fg]
      - box regressor         -> (N, 2*4) offsets
    """

    def __init__(self, in_channels: int, num_classes: int = 2, nhead: int = 8):
        super().__init__()
        self.d_model = in_channels

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=False,  # (seq, batch, dim)
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        self.cls_score = nn.Linear(self.d_model, 1)
        self.bbox_pred = nn.Linear(self.d_model, num_classes * 4)

        self.prototype_tokens: torch.Tensor | None = None  # (K, D)

    def set_prototype_tokens(self, proto_tokens: torch.Tensor):
        """
        proto_tokens: (K,D) or (1,D)
        """
        if proto_tokens.dim() == 1:
            proto_tokens = proto_tokens.unsqueeze(0)
        self.prototype_tokens = F.normalize(proto_tokens, dim=-1)

    def forward(self, x: torch.Tensor):
        """
        x: (N, D) RoI features from box_head
        """
        if self.prototype_tokens is None:
            raise ValueError(
                "Prototype not set in CrossAttentionSimilarityHead. "
                "Call set_prototype_tokens() before forward()."
            )

        x = F.normalize(x, dim=-1)                          # (N, D)
        proto = F.normalize(self.prototype_tokens, dim=-1)  # (K, D)

        q = x.unsqueeze(1)          # (N,1,D)
        k = proto.unsqueeze(1)      # (K,1,D)
        v = k

        att_out, _ = self.cross_attn(q, k, v)  # (N,1,D)
        att_out = att_out.squeeze(1)           # (N,D)

        feat = x + self.ffn(att_out)
        # classifier
        sim = self.cls_score(feat).squeeze(-1)  # (N,)
        scores = torch.stack([-sim, sim], dim=1)
        # bbox regressor
        bbox_deltas = self.bbox_pred(feat)
        return scores, bbox_deltas


# =====================================================
#  4) MODEL BUILDER
# =====================================================

def build_siamese_model(agnostic_weights=None):
    """
    Next-gen Siamese FSOD model:

      - FasterRCNN backbone + FPN
      - Support-Conditioned RPN
      - Cross-Attention RoI head
      - PrototypeEncoder (external, used on supports)
    """
    model = tv.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    roi_feat_dim = model.roi_heads.box_head.fc7.out_features  # D

    # --- support-conditioned RPN ---
    rpn_in_channels = model.backbone.out_channels
    num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
    model.rpn.head = SupportConditionedRPNHead(
        in_channels=rpn_in_channels,
        num_anchors=num_anchors,
        proto_dim=roi_feat_dim,
    )

    # --- cross-attention RoI predictor ---
    model.roi_heads.box_predictor = CrossAttentionSimilarityHead(
        in_channels=roi_feat_dim,
        num_classes=2,
        nhead=8,
    )

    if agnostic_weights:
        print(f"Loading backbone/RPN weights from: {agnostic_weights}")
        ckpt = torch.load(agnostic_weights, map_location="cpu")["model"]

        # Drop old detection heads (RPN + box_predictor)
        keys_to_remove = [
            k for k in ckpt.keys()
            if k.startswith("roi_heads.box_predictor.")
            or k.startswith("rpn.head.")
        ]
        for k in keys_to_remove:
            del ckpt[k]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print("Loaded pretrain. Missing:", missing)
        print("Unexpected:", unexpected)

    return model


# =====================================================
#  5) PROTOTYPE BUILDING FROM SUPPORT IMAGES
# =====================================================

@torch.no_grad()
def encode_support_prototypes(model, proto_encoder, support_tensors, device):
    """
    support_tensors: (K,C,H,W) normalized

    Steps:
      - run K images through backbone+RoI head
      - get K feature vectors (K,D)
      - pass through PrototypeEncoder
      - return proto_tokens (K,D), proto_global (D,)
    """
    model.eval()
    proto_encoder.eval()

    supp = support_tensors.to(device).float()
    K = supp.shape[0]
    feats_all = []

    for i in range(K):
        img = supp[i:i + 1]  # (1,C,H,W)
        feats = model.backbone(img)
        _, _, H, W = img.shape
        boxes = [torch.tensor([[0., 0., W - 1, H - 1]], device=device)]
        shapes = [(H, W)]

        pooled = model.roi_heads.box_roi_pool(feats, boxes, shapes)
        feat = model.roi_heads.box_head(pooled).squeeze(0)  # (D,)
        feats_all.append(feat)

    support_feats = torch.stack(feats_all, dim=0)  # (K,D)
    proto_tokens, proto_global = proto_encoder(support_feats)
    return proto_tokens, proto_global


# =====================================================
#  6) UTILS
# =====================================================

def denormalize_for_model(img_tensor, device):
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


# =====================================================
#  7) TRAIN ONE EPISODE
# =====================================================

def train_one_episode(model, proto_encoder, episode_batch, optimizer, device, max_norm=1.0):
    supports, query_img, target = episode_batch
    supports = supports[0].to(device)
    query_img = query_img[0].to(device)

    tgt = {
        "boxes": target["boxes"][0].to(device),
        "labels": target["labels"][0].to(device),
    }

    # ---- 1. encode prototypes from supports ----
    with torch.no_grad():
        proto_tokens, proto_global = encode_support_prototypes(
            model, proto_encoder, supports, device
        )

    # set prototypes in heads
    model.roi_heads.box_predictor.set_prototype_tokens(proto_tokens)
    if hasattr(model.rpn.head, "set_prototype_global"):
        model.rpn.head.set_prototype_global(proto_global)

    # ---- 2. forward + detection loss ----
    query_img_0_1 = denormalize_for_model(query_img, device)
    model.train()
    loss_dict = model([query_img_0_1], [tgt])
    loss_det = sum(loss for loss in loss_dict.values())

    # ---- 3. episodic recall (for logging) ----
    model.eval()
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

    model.train()

    # ---- 4. backward ----
    optimizer.zero_grad(set_to_none=True)
    loss_det.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()

    # clear prototypes
    model.roi_heads.box_predictor.prototype_tokens = None
    if hasattr(model.rpn.head, "clear_prototype"):
        model.rpn.head.clear_prototype()

    loss_det_val = float(loss_det.detach().item())
    loss_dict_detached = {k: float(v.detach().item()) for k, v in loss_dict.items()}
    loss_dict_detached["loss"] = loss_det_val
    return loss_dict_detached, ep_recall


# =====================================================
#  8) EVALUATION (AP / PRECISION / RECALL)
# =====================================================

@torch.no_grad()
def evaluate_siamese(model, proto_encoder, val_loader, device, iou_thresh=0.5):
    model.eval()
    proto_encoder.eval()

    all_scores = []
    all_tp_flags = []
    total_gt = 0

    total_hit_thr = 0
    total_pred_thr = 0

    for supports, query_img, target in tqdm(val_loader, desc="Validating"):
        supports = supports[0].to(device)
        query_img = query_img[0].to(device)
        boxes_gt = target["boxes"][0].to(device)
        labels_gt = target["labels"][0].to(device)

        pos_mask = labels_gt == 1
        boxes_gt = boxes_gt[pos_mask]
        num_gt = boxes_gt.shape[0]
        total_gt += num_gt

        proto_tokens, proto_global = encode_support_prototypes(
            model, proto_encoder, supports, device
        )
        model.roi_heads.box_predictor.set_prototype_tokens(proto_tokens)
        if hasattr(model.rpn.head, "set_prototype_global"):
            model.rpn.head.set_prototype_global(proto_global)

        query_img_0_1 = denormalize_for_model(query_img, device)
        preds = model([query_img_0_1])[0]

        model.roi_heads.box_predictor.prototype_tokens = None
        if hasattr(model.rpn.head, "clear_prototype"):
            model.rpn.head.clear_prototype()

        if num_gt == 0 and len(preds["boxes"]) == 0:
            continue

        fg_mask = preds["labels"] == 1
        boxes_pred = preds["boxes"][fg_mask].to(device)
        scores_pred = preds["scores"][fg_mask].to(device)

        # --- PR curve ---
        if num_gt > 0 and boxes_pred.numel() > 0:
            ious = box_iou(boxes_pred, boxes_gt)
            best_ious, best_gt_idx = ious.max(dim=1)
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
            for s in scores_pred:
                all_scores.append(float(s))
                all_tp_flags.append(0)

        # --- fixed threshold 0.5 ---
        keep_thr = scores_pred >= 0.5
        boxes_pred_thr = boxes_pred[keep_thr]
        total_pred_thr += boxes_pred_thr.shape[0]

        if num_gt > 0 and boxes_pred_thr.numel() > 0:
            ious_thr = box_iou(boxes_pred_thr, boxes_gt)
            hits_thr = (ious_thr.max(dim=0).values >= iou_thresh).sum().item()
            total_hit_thr += hits_thr

    if len(all_scores) == 0:
        return {"ap": 0.0, "pr_curve": ([], []), "recall": 0.0, "precision": 0.0}

    scores_np = np.array(all_scores)
    tps_np = np.array(all_tp_flags, dtype=np.int32)
    order = np.argsort(-scores_np)
    tps_sorted = tps_np[order]
    fps_sorted = 1 - tps_sorted

    cum_tps = np.cumsum(tps_sorted)
    cum_fps = np.cumsum(fps_sorted)

    recalls = cum_tps / max(total_gt, 1)
    precisions = cum_tps / np.maximum(cum_tps + cum_fps, 1e-8)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(1, mrec.size):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    recall_thr = total_hit_thr / max(total_gt, 1)
    precision_thr = total_hit_thr / max(total_pred_thr, 1)

    return {
        "ap": float(ap),
        "pr_curve": (recalls.tolist(), precisions.tolist()),
        "recall": float(recall_thr),
        "precision": float(precision_thr),
    }


# =====================================================
#  9) OPTIMIZER
# =====================================================

def build_optimizer(model, proto_encoder, lr_head, backbone_lr_scale, freeze_backbone=False):
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.roi_heads.box_predictor.parameters()) + \
                  list(model.rpn.head.parameters()) + \
                  list(proto_encoder.parameters())

    param_groups = [
        {"params": [p for p in head_params if p.requires_grad], "lr": lr_head}
    ]

    if not freeze_backbone:
        param_groups.append(
            {
                "params": [p for p in backbone_params if p.requires_grad],
                "lr": lr_head * backbone_lr_scale,
            }
        )
    else:
        print("Freezing backbone parameters.")
        for p in backbone_params:
            p.requires_grad = False

    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
    if not param_groups:
        raise ValueError("No parameters to optimize. Check if model is frozen.")
    return optim.AdamW(param_groups, lr=lr_head, weight_decay=1e-4)


# =====================================================
#  10) MAIN
# =====================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--agnostic_weights", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--backbone_lr_scale", type=float, default=0.1)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--unfreeze_at_epoch", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="checkpoints_siamese_nextgen")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume_from", type=str, default=None)

    args = ap.parse_args()

    loaders = make_loaders(
        data_root=Path(args.data_root),
        annot_path=Path(args.data_root) / "annotations/annotations.json",
        samples_root=Path(args.data_root) / "samples",
        batch_size_train=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = loaders["train_episode_loader"]
    val_loader = loaders["val_episode_loader"]

    device = torch.device(args.device)

    model = build_siamese_model(args.agnostic_weights).to(device)
    roi_feat_dim = model.roi_heads.box_head.fc7.out_features
    proto_encoder = PrototypeEncoder(d_model=roi_feat_dim, nhead=4).to(device)

    start_epoch = 1
    best_recall = -1.0
    best_path = None

    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "proto_encoder" in ckpt:
            proto_encoder.load_state_dict(ckpt["proto_encoder"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("val_recall", -1.0)
        best_path = args.resume_from

    os.makedirs(args.out_dir, exist_ok=True)

    current_freeze_state = args.freeze_backbone
    if args.freeze_backbone and start_epoch > args.unfreeze_at_epoch:
        print("Resume after unfreeze epoch; starting with backbone unfrozen.")
        current_freeze_state = False

    optimizer = build_optimizer(
        model, proto_encoder,
        lr_head=args.lr_head,
        backbone_lr_scale=args.backbone_lr_scale,
        freeze_backbone=current_freeze_state,
    )

    print("Starting NEXT-GEN Siamese episodic training...")

    for epoch in range(start_epoch, args.epochs + 1):
        # simple negative-prob curriculum: ramp up over epochs
        neg_prob = min(0.4, 0.05 * (epoch - 1))  # e.g. 0.0, 0.05, ... 0.4
        if hasattr(train_loader.dataset, "negative_prob"):
            train_loader.dataset.negative_prob = neg_prob
        print(f"Epoch {epoch}: negative_prob = {neg_prob:.2f}")

        if current_freeze_state and epoch == args.unfreeze_at_epoch:
            print(f"Unfreezing backbone at epoch {epoch}")
            optimizer = build_optimizer(
                model, proto_encoder,
                lr_head=args.lr_head,
                backbone_lr_scale=args.backbone_lr_scale,
                freeze_backbone=False,
            )
            current_freeze_state = False

        model.train()
        proto_encoder.train()
        running_loss = 0.0
        running_recall = 0.0

        for i, episode_batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} train")):
            loss_dict, ep_recall = train_one_episode(
                model, proto_encoder, episode_batch, optimizer, device
            )
            running_loss += loss_dict["loss"]
            running_recall += ep_recall

            if (i + 1) % 50 == 0:
                print(
                    f"  Iter {i + 1}/{len(train_loader)} "
                    f"| avg_loss={running_loss / (i + 1):.4f} "
                    f"| avg_recall={running_recall / (i + 1):.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        avg_recall = running_recall / len(train_loader)
        print(f"[Epoch {epoch}] Train done. Avg_loss={avg_loss:.4f}, Avg_recall={avg_recall:.4f}")

        metrics = evaluate_siamese(model, proto_encoder, val_loader, device)
        print(
            f"[Epoch {epoch}] Val AP={metrics['ap']:.4f}, "
            f"Recall@0.5={metrics['recall']:.4f}, "
            f"Precision@0.5={metrics['precision']:.4f}"
        )
        val_recall = metrics["recall"]

        ckpt_path = os.path.join(args.out_dir, f"siamese_nextgen_epoch{epoch:02d}.pth")
        torch.save(
            {
                "model": model.state_dict(),
                "proto_encoder": proto_encoder.state_dict(),
                "epoch": epoch,
                "val_recall": val_recall,
            },
            ckpt_path,
        )

        if val_recall > best_recall:
            best_recall = val_recall
            best_path = ckpt_path
            print(f"  -> New best recall saved at {best_path}")

    print(f"\nTraining done. Best recall={best_recall:.4f} @ {best_path}")


if __name__ == "__main__":
    main()
