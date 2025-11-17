#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv

# Reuse your data utilities
try:
    from Data import make_loaders, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    raise SystemExit("Please place data_setup.py in the same folder or PYTHONPATH.")

# albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_val_transforms(img_size=800):
    """
    Val transforms: resize longest side to img_size, pad to square img_size x img_size,
    then ToTensorV2 so output is C,H,W in float [0..1].
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
    )


# --- modified apply_val_transform ---
def apply_val_transform(img_tensor, transform, mean, std, device, bboxes=None, labels=None):
    """
    img_tensor: torch.Tensor C,H,W normalized by (img - mean)/std
    bboxes: list/ndarray/Tensor of shape (N,4) in absolute xyxy coords (optional)
    returns: (img_t_tensor, transformed_bboxes_list)
      - img_t_tensor: torch.Tensor C,H,W float [0..1] on `device`
      - transformed_bboxes_list: list of [x_min,y_min,x_max,y_max] in absolute coords on transformed image
    """
    img_cpu = img_tensor.detach().cpu()
    mean_t = torch.tensor(mean).view(3,1,1)
    std_t = torch.tensor(std).view(3,1,1)
    img_denorm = (img_cpu * std_t + mean_t).permute(1,2,0).numpy().astype("float32")  # H,W,C

    # prepare alb inputs
    alb_kwargs = {"image": img_denorm}
    if bboxes is not None:
        # albumentations wants list of boxes and labels
        if isinstance(bboxes, torch.Tensor):
            bboxes_list = bboxes.detach().cpu().tolist()
        else:
            bboxes_list = list(map(list, bboxes))
        labels_list = labels.detach().cpu().tolist() if (labels is not None and isinstance(labels, torch.Tensor)) else (labels or [])
        alb_kwargs["bboxes"] = bboxes_list
        alb_kwargs["labels"] = labels_list
    else:
        alb_kwargs["bboxes"] = []
        alb_kwargs["labels"] = []

    transformed = transform(**alb_kwargs)
    img_t = transformed["image"]  # C,H,W in 0..1 (ToTensorV2)
    transformed_bboxes = transformed.get("bboxes", [])

    # move image to device
    return img_t.to(device), transformed_bboxes


# --- updated make_support_proto (if you need supports resized) ---
@torch.no_grad()
def make_support_proto(model, support_tensors, device, val_transform):
    model.eval()
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    feats_all = []
    for i in range(support_tensors.shape[0]):
        img_t, _ = apply_val_transform(support_tensors[i], val_transform, mean, std, device)
        img_b = img_t.unsqueeze(0)
        features = model.backbone(img_b)
        _, _, H, W = img_b.shape
        boxes = [torch.tensor([[0., 0., float(W - 1), float(H - 1)]], device=device)]
        image_shapes = [(H, W)]
        pooled = model.roi_heads.box_roi_pool(features, boxes, image_shapes)
        feat = model.roi_heads.box_head(pooled)
        feats_all.append(F.normalize(feat.squeeze(0), dim=-1))
    proto = F.normalize(torch.stack(feats_all, dim=0).mean(dim=0), dim=-1)
    return proto


# --- updated episode_infer: now transforms query AND returns transformed GT boxes ---
@torch.no_grad()
def episode_infer(model, support_tensors, query_tensor, gt_boxes_orig, gt_labels_orig, device, val_transform, score_thr=0.3, cos_thr=0.6, topk=200, debug=False):
    proto = make_support_proto(model, support_tensors, device, val_transform)

    # Transform query AND its GT boxes so coordinates align with model inputs
    query_t, transformed_gt_boxes = apply_val_transform(query_tensor, val_transform, IMAGENET_MEAN, IMAGENET_STD, device, bboxes=gt_boxes_orig, labels=gt_labels_orig)
    query_b = query_t.unsqueeze(0)  # 1,C,H,W

    # Raw preds
    pred = model([query_b[0]])[0]
    if debug:
        print("  raw preds:", len(pred.get("boxes", [])), "scores top10:", pred.get("scores", [])[:10] if "scores" in pred else None, "labels top10:", pred.get("labels", [])[:10] if "labels" in pred else None)

    keep = pred["scores"] >= score_thr
    boxes = pred["boxes"][keep][:topk]
    scores = pred["scores"][keep][:topk]

    if boxes.numel() == 0:
        return {"boxes": boxes, "scores": scores, "cos": torch.empty(0, device=device), "gt_transformed": transformed_gt_boxes}

    features = model.backbone(query_b)
    image_shapes = [query_b.shape[-2:]]
    pooled = model.roi_heads.box_roi_pool(features, [boxes], image_shapes)
    roi_feats = model.roi_heads.box_head(pooled)
    roi_feats = F.normalize(roi_feats, dim=-1)
    cos = (roi_feats @ proto)

    if debug:
        print("  cos stats: min", float(cos.min().item()), "max", float(cos.max().item()), "mean", float(cos.mean().item()))

    mask = cos >= cos_thr
    return {
        "boxes": boxes[mask],
        "scores": scores[mask],
        "cos": cos[mask],
        "gt_transformed": transformed_gt_boxes
    }



@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="path to trained model .pth from train_frcnn.py")
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--novel_labels", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--cos_thr", type=float, default=0.6)
    ap.add_argument("--img_size", type=int, default=800, help="validation transform size (longest side and pad)")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Build model
    model = tv.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = tv.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    # val transform
    val_transform = get_val_transforms(img_size=args.img_size)

    # Build episode loader
    novel = [s for s in args.novel_labels.split(",") if s] if args.novel_labels else None
    loaders = make_loaders(
        data_root=Path(args.data_root),
        annot_path=Path(args.data_root) / "annotations" / "annotations.json",
        samples_root=Path(args.data_root) / "samples",
        novel_labels=novel,
        batch_size_train=4,
        num_workers=4,
    )
    ep_loader = loaders["val_episode_loader"] if len(loaders.get("val_episode_loader", [])) else loaders["test_episode_loader"]

    # Simple pass over episodes and print counts
    from torchvision.ops import box_iou
    total = 0
    matched = 0
    for i, (supports, query, target) in enumerate(ep_loader):
        # supports and query as before; ensure shapes align
        supports_t = supports[0] if supports.dim() == 5 else supports
        query_t = query[0] if query.dim() == 4 else query
        gt_boxes_orig = target["boxes"][0]  # original GT boxes (absolute coords)
        gt_labels_orig = target["labels"][0]

        det = episode_infer(model, supports_t, query_t, gt_boxes_orig, gt_labels_orig, device, val_transform,
                            score_thr=args.score_thr, cos_thr=args.cos_thr, debug=True)
        gt_transformed = det["gt_transformed"]
        # convert gt_transformed (list) to tensor on device for IoU comparison
        if len(gt_transformed) > 0:
            gt_transformed_t = torch.tensor(gt_transformed, dtype=torch.float32, device=device)
        else:
            gt_transformed_t = torch.zeros((0, 4), dtype=torch.float32, device=device)

        total += gt_transformed_t.shape[0]
        if gt_transformed_t.numel() and det["boxes"].numel():
            iou = box_iou(det["boxes"], gt_transformed_t)
            matched += (iou.max(dim=0).values >= 0.5).sum().item()

        print(f"Episode {i}: kept {det['boxes'].shape[0]} boxes (score>={args.score_thr}, cos>={args.cos_thr})")
        if i >= 19:  # limit default output
            break
    if total > 0:
        print(f"Quick Recall@0.5 over {min(20, len(ep_loader))} episodes: {matched/total:.4f}")
    else:
        print("No GT boxes found in sampled episodes.")


if __name__ == "__main__":
    main()
