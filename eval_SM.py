import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.ops import box_iou

# We reuse the data utilities you already have:
try:
    from Data import make_loaders, IMAGENET_MEAN, IMAGENET_STD
except ImportError as e:
    raise SystemExit("Please place Data.py in the same folder or in PYTHONPATH.")


# -----------------------------------------------------------------------------
# 1. MODEL DEFINITIONS (Copied from siamese.py)
# -----------------------------------------------------------------------------

class SimilarityHead(nn.Module):
    """
    A stateful box predictor that compares RoI features to a
    set prototype. Replaces the standard FastRCNNPredictor.
    """

    def __init__(self, in_channels, proto_channels, num_classes=2):
        super().__init__()
        combined_dim = in_channels + proto_channels

        # Binary classifier: 1 = match, 0 = background
        self.cls_score = nn.Linear(combined_dim, num_classes)
        # Bbox regressor (must be num_classes * 4)
        self.bbox_pred = nn.Linear(combined_dim, num_classes * 4)

        self.prototype = None

    def set_prototype(self, proto):
        """Sets the (1024-D) prototype for the current episode."""
        self.prototype = proto

    def forward(self, x):
        """x: (N, D_in) RoI features (e.g., [N, 1024])"""
        if self.prototype is None:
            raise ValueError("Prototype not set in SimilarityHead. Call set_prototype() first.")

        proto_expanded = self.prototype.unsqueeze(0).expand(x.shape[0], -1)
        combined_features = torch.cat([x, proto_expanded], dim=1)

        if combined_features.dim() == 4:
            combined_features = combined_features.flatten(start_dim=1)

        scores = self.cls_score(combined_features)
        bbox_deltas = self.bbox_pred(combined_features)
        return scores, bbox_deltas


def build_siamese_model(agnostic_weights_path=None):
    """Builds the Siamese model for Step 7."""
    model = tv.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)

    # Get feature dimensions (e.g., 1024)
    roi_feat_dim = model.roi_heads.box_head.fc7.out_features

    # Replace the final predictor
    model.roi_heads.box_predictor = SimilarityHead(
        in_channels=roi_feat_dim,
        proto_channels=roi_feat_dim,
        num_classes=2
    )

    if agnostic_weights_path:
        # This part is used for training, not eval, but we keep the
        # function signature the same for consistency.
        print(f"Loading weights from: {agnostic_weights_path}")
        ckpt = torch.load(agnostic_weights_path, map_location="cpu")["model"]

        keys_to_remove = [k for k in ckpt.keys() if k.startswith("roi_heads.box_predictor.")]
        for k in keys_to_remove:
            del ckpt[k]

        model.load_state_dict(ckpt, strict=False)

    return model


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Copied from siamese.py)
# -----------------------------------------------------------------------------

@torch.no_grad()
def make_support_proto(model, support_tensors, device):
    """
    Creates the L2-normalized prototype vector.
    support_tensors: (3, C, H, W), *already normalized*
    """
    model.eval()  # Use eval mode for prototype creation
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # Denormalize to [0,1] for the model's internal transform
    supp = (support_tensors.to(device).float() * std + mean).clamp(0, 1)

    feats_all = []
    for i in range(supp.shape[0]):
        img = supp[i:i + 1]  # (1,C,H,W)
        features = model.backbone(img)
        _, _, H, W = img.shape
        boxes = [torch.tensor([[0., 0., W - 1., H - 1.]], device=device)]
        image_shapes = [(H, W)]

        pooled = model.roi_heads.box_roi_pool(features, boxes, image_shapes)
        feat = model.roi_heads.box_head(pooled)  # (1, 1024)
        feats_all.append(F.normalize(feat.squeeze(0), dim=-1))

    proto = F.normalize(torch.stack(feats_all, dim=0).mean(dim=0), dim=-1)  # (1024,)
    return proto


def denormalize_for_model(img_tensor, device):
    """Denormalizes a single C,H,W tensor back to [0,1] for model input."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


# -----------------------------------------------------------------------------
# 3. EVALUATION FUNCTION (Copied from siamese.py)
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_siamese(model, val_episode_loader, device, score_thresh=0.5, iou_thresh=0.5):
    """
    Runs full episodic evaluation on the validation set.
    """
    model.eval()
    total_gt = 0
    total_hit = 0

    for i, (supports, query_img, target) in enumerate(val_episode_loader):
        # --- 1. Get data (batch_size=1) ---
        supports = supports[0].to(device)
        query_img = query_img[0].to(device)
        boxes_gt = target["boxes"][0].to(device)
        total_gt += boxes_gt.shape[0]

        # --- 2. Create and set prototype ---
        proto = make_support_proto(model, supports, device)
        model.roi_heads.box_predictor.set_prototype(proto)

        # --- 3. Run inference ---
        query_img_0_1 = denormalize_for_model(query_img, device)
        preds = model([query_img_0_1])[0]

        model.roi_heads.box_predictor.prototype = None  # Clear prototype

        # --- 4. Filter & Compare ---
        # The model's class 1 score *is* the similarity score.
        keep = preds["labels"] == 1  # Keep only "match" class
        keep = keep & (preds["scores"] >= score_thresh)
        boxes_pred = preds["boxes"][keep]

        if boxes_gt.numel() == 0 or boxes_pred.numel() == 0:
            continue

        ious = box_iou(boxes_pred, boxes_gt)  # [Np, Ng]
        hits = (ious.max(dim=0).values >= iou_thresh).sum().item()
        total_hit += hits

        if (i + 1) % 20 == 0:
            print(f"  ...processed {i + 1} / {len(val_episode_loader)} episodes")

    recall = (total_hit / total_gt) if total_gt > 0 else 0.0
    return recall


# -----------------------------------------------------------------------------
# 4. MAIN SCRIPT LOGIC
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to your best *Siamese* model .pth")
    ap.add_argument("--data_root", type=str, default=".", help="root containing annotations/ and samples/")
    ap.add_argument("--novel_labels", type=str, required=True, help="comma-separated class names to treat as novel")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_thr", type=float, default=0.5, help="Confidence threshold for a detection to be a 'match'")
    args = ap.parse_args()

    # Build loaders
    novel = [s for s in args.novel_labels.split(",") if s]
    loaders = make_loaders(
        data_root=Path(args.data_root),
        annot_path=Path(args.data_root) / "annotations" / "annotations.json",
        samples_root=Path(args.data_root) / "samples",
        novel_labels=novel,
        batch_size_train=1,  # Not used, but required
        num_workers=4,
    )

    # --- Use the TEST loader ---
    test_ep_loader = loaders["test_episode_loader"]
    if len(test_ep_loader) == 0:
        print("Error: No test episodes found. Did you provide the correct novel labels?")
        return

    device = torch.device(args.device)

    # 1. Build Model (with None weights)
    model = build_siamese_model(agnostic_weights_path=None).to(device)

    # 2. Load your trained checkpoint
    print(f"Loading Siamese checkpoint from: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Evaluating on {len(test_ep_loader)} novel test episodes...")
    recall = evaluate_siamese(
        model,
        test_ep_loader,
        device,
        score_thresh=args.score_thr
    )

    print("\n--- Final Results (Novel Classes) ---")
    print(f"Score Threshold: {args.score_thr}")
    print(f"Recall@0.5IoU:   {recall:.4f}")


if __name__ == "__main__":
    main()