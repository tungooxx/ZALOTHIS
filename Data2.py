# Data.py
import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

# ---------- CONFIG ----------
#    DATA_ROOT/
#     annotations/annotations.json
#     samples/<ClassN>_<idx>/{drone_video.mp4, object_images/img_1.jpg..img_3.jpg}

DATA_ROOT = Path(".")
ANNOT_PATH = DATA_ROOT / "annotations" / "annotations.json"
SAMPLES_ROOT = DATA_ROOT / "samples"

# torchvision detection normalization (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------
# Utilities
# --------------------------------------------
def parse_label_and_index(video_id: str) -> Tuple[str, Optional[int]]:
    if "_" in video_id:
        name, idx = video_id.rsplit("_", 1)
        try:
            return name, int(idx)
        except ValueError:
            return name, None
    return video_id, None


def load_annotations(annot_path: Path) -> Dict[str, Dict[int, List[List[int]]]]:
    """
    JSON format:
    [
      {
        "video_id": "Backpack_0",
        "annotations": [
          { "bboxes": [ {"frame": 3483, "x1":..,"y1":..,"x2":..,"y2":..}, ... ] }
        ]
      }
    ]
    """
    with open(annot_path, "r") as f:
        data = json.load(f)

    by_video: Dict[str, Dict[int, List[List[int]]]] = {}
    for item in data:
        vid = item["video_id"]
        frames: Dict[int, List[List[int]]] = by_video.setdefault(vid, {})
        for seg in item.get("annotations", []):
            for bb in seg.get("bboxes", []):
                fr = int(bb["frame"])
                box = [int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])]
                frames.setdefault(fr, []).append(box)
    return by_video


def discover_samples(samples_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Map video_id -> {
        "video_path": Path,
        "support_paths": [Path, Path, Path],
        "class_name": str
    }
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not samples_root.exists():
        return out
    for d in sorted(samples_root.iterdir()):
        if not d.is_dir():
            continue
        video_id = d.name
        video_path = d / "drone_video.mp4"
        supp_dir = d / "object_images"
        supports = [supp_dir / f"img_{i}.jpg" for i in (1, 2, 3)]
        if not video_path.exists() or not all(p.exists() for p in supports):
            continue
        cls, _ = parse_label_and_index(video_id)
        out[video_id] = {
            "video_path": video_path,
            "support_paths": supports,
            "class_name": cls,
        }
    return out


def cv2_read_frame_at(video_path: Path, frame_idx: int) -> np.ndarray:
    """Return RGB uint8 HxWx3 for a specific 0-based frame index."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def to_tensor_and_normalize(img_rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 -> float tensor CxHxW in [0,1], normalized by ImageNet stats."""
    img = torch.from_numpy(img_rgb).contiguous().float() / 255.0  # HWC
    img = img.permute(2, 0, 1)  # CHW
    return F.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def resize_keep_aspect(img: np.ndarray, short_side: int = 800, long_cap: int = 1333) -> Tuple[np.ndarray, float]:
    """Uniformly scale so min(H,W)=short_side (cap max to long_cap). Return (img, scale)."""
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if max(h, w) * scale > long_cap:
        scale = long_cap / max(h, w)
    if abs(scale - 1.0) > 1e-6:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img, scale


def scale_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    return boxes * scale


def collate_fn_detection(batch):
    # For torchvision detection models: return lists
    imgs, targets = zip(*batch)  # tuples of length B
    return list(imgs), list(targets)


# --------------------------------------------
# Split builder
# --------------------------------------------
def build_splits(
        ann_by_video: Dict[str, Dict[int, List[List[int]]]],
        sample_info: Dict[str, Dict[str, Any]],
        novel_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    by_class: Dict[str, List[str]] = {}
    for vid in ann_by_video.keys():
        cls, _ = parse_label_and_index(vid)
        by_class.setdefault(cls, []).append(vid)

    classes = sorted(by_class.keys())
    if novel_labels is None:
        novel_labels = classes[-2:] if len(classes) >= 2 else []
    base_labels = [c for c in classes if c not in set(novel_labels)]

    def is_base(vid: str) -> bool:
        cls, _ = parse_label_and_index(vid)
        return cls in base_labels

    def is_novel(vid: str) -> bool:
        cls, _ = parse_label_and_index(vid)
        return cls in novel_labels

    train_samples: List[Tuple[str, int, np.ndarray]] = []
    val_episodes: List[Tuple[str, int]] = []
    test_episodes: List[Tuple[str, int]] = []

    for vid, frames_dict in ann_by_video.items():
        if vid not in sample_info:
            continue
        cls, idx = parse_label_and_index(vid)
        frame_ids = sorted(frames_dict.keys())

        if is_base(vid):
            if idx is not None:
                if idx == 0:  # Use <Class>_0 for training
                    for fr in frame_ids:
                        boxes = np.array(frames_dict[fr], dtype=np.float32)
                        train_samples.append((vid, fr, boxes))
                else:  # Use <Class>_1, _2 etc. for validation
                    for fr in frame_ids:
                        val_episodes.append((vid, fr))
            else:
                # No _idx, so split frames 80/20
                n = len(frame_ids)
                n_train = int(math.ceil(0.8 * n))
                for fr in frame_ids[:n_train]:
                    boxes = np.array(frames_dict[fr], dtype=np.float32)
                    train_samples.append((vid, fr, boxes))
                for fr in frame_ids[n_train:]:
                    val_episodes.append((vid, fr))
        elif is_novel(vid):
            for fr in frame_ids:
                test_episodes.append((vid, fr))

    # --- NEW ---
    # Create an episodes list from the training samples for Step 7
    train_episodes = [(vid, fr) for vid, fr, _ in train_samples]
    # --- END NEW ---

    return {
        "train_samples": train_samples,
        "train_episodes": train_episodes,  # <-- NEW
        "val_episodes": val_episodes,
        "test_episodes": test_episodes,
        "base_labels": base_labels,
        "novel_labels": novel_labels,
    }


# --------------------------------------------
# Datasets
# --------------------------------------------
class DetectionTrainDataset(Dataset):
    """Class-agnostic detection dataset for torchvision models."""

    def __init__(
            self,
            train_samples: List[Tuple[str, int, np.ndarray]],
            sample_info: Dict[str, Dict[str, Any]],
            resize_short: int = 800,
            resize_long_cap: int = 1333,
            hflip_p: float = 0.5,
    ):
        self.samples = train_samples
        self.info = sample_info
        self.resize_short = resize_short
        self.resize_long_cap = resize_long_cap
        self.hflip_p = hflip_p
        self.rng = np.random.RandomState(1337)

    def __len__(self) -> int:
        return len(self.samples)

    def _hflip(self, img: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        img_flipped = cv2.flip(img, 1)
        if boxes.size == 0:
            return img_flipped, boxes
        x1 = boxes[:, 0].copy()
        x2 = boxes[:, 2].copy()
        boxes[:, 0] = w - x2 - 1
        boxes[:, 2] = w - x1 - 1
        return img_flipped, boxes

    def __getitem__(self, idx: int):
        vid, fr, boxes = self.samples[idx]
        video_path = self.info[vid]["video_path"]
        img = cv2_read_frame_at(video_path, fr)  # RGB

        if self.rng.rand() < self.hflip_p:
            img, boxes = self._hflip(img, boxes)

        img, scale = resize_keep_aspect(img, self.resize_short, self.resize_long_cap)
        if boxes.size > 0:
            boxes = scale_boxes(boxes, scale)

        img_t = to_tensor_and_normalize(img)  # CPU tensor
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),  # CPU
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),  # CPU
            "image_id": torch.tensor(idx, dtype=torch.int64),  # scalar CPU
        }
        return img_t, target


class EpisodeDataset(Dataset):
    """
    Episodes:
      return (supports: (3,C,Hs,Ws), query: (C,H,W), target dict)

    train_mode=True:
      - With probability `negative_prob`, create a NEGATIVE episode:
          support class != query class, all labels = 0
      - Otherwise POSITIVE episode:
          support class == query class, labels = 1
    """

    def __init__(
            self,
            episodes: List[Tuple[str, int]],
            ann_by_video: Dict[str, Dict[int, List[List[int]]]],
            sample_info: Dict[str, Dict[str, Any]],
            support_size: int = 256,
            query_short: int = 800,
            query_long_cap: int = 1333,
            train_mode: bool = False,
            negative_prob: float = 0.6,   # <--- stronger negatives
    ):
        self.episodes = episodes
        self.ann = ann_by_video
        self.info = sample_info
        self.support_size = support_size
        self.query_short = query_short
        self.query_long_cap = query_long_cap

        self.train_mode = train_mode
        self.negative_prob = negative_prob

        if train_mode:
            self.rng = np.random.RandomState(1337)
            # Map: class_name -> [video_ids]
            self.vids_by_class: Dict[str, List[str]] = {}
            for vid, info in self.info.items():
                cls = info["class_name"]
                self.vids_by_class.setdefault(cls, []).append(vid)
            self.all_classes = sorted(self.vids_by_class.keys())

    def __len__(self) -> int:
        return len(self.episodes)

    def _read_and_preprocess_supports(self, support_paths: List[Path]) -> torch.Tensor:
        """
        Read support images, apply a light central crop (to remove some background)
        and resize to `support_size`. This makes prototypes more focused on the object.
        """
        supp_tensors = []
        for p in support_paths:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                raise RuntimeError(f"Cannot read support image: {p}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # ---- central crop to reduce background (Fix 3) ----
            h, w = img.shape[:2]
            crop_ratio = 0.8  # keep 80% of the shorter side
            ch = int(h * crop_ratio)
            cw = int(w * crop_ratio)
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            img = img[y0:y0 + ch, x0:x0 + cw]

            img = cv2.resize(img, (self.support_size, self.support_size),
                             interpolation=cv2.INTER_LINEAR)
            supp_tensors.append(to_tensor_and_normalize(img))  # CPU
        return torch.stack(supp_tensors, dim=0)  # (K, C, H, W)

    def __getitem__(self, idx: int):
        # 1. Pick the query frame from the predefined episode list
        vid_query, fr = self.episodes[idx]
        vinfo_query = self.info[vid_query]
        video_path_query = vinfo_query["video_path"]
        class_query = vinfo_query["class_name"]

        # Decide POSITIVE vs NEGATIVE episode
        make_negative_episode = False
        if self.train_mode and self.rng.rand() < self.negative_prob:
            make_negative_episode = True

        # 2. Choose support class and video
        if make_negative_episode:
            # choose a different class
            possible_neg_classes = [c for c in self.all_classes if c != class_query]
            if not possible_neg_classes:
                # only one class exists, fallback to positive
                make_negative_episode = False
                class_support = class_query
            else:
                class_support = self.rng.choice(possible_neg_classes)
        else:
            class_support = class_query

        if self.train_mode:
            vid_support = self.rng.choice(self.vids_by_class[class_support])
        else:
            # val/test: support and query from the same video/class
            vid_support = vid_query

        support_paths = self.info[vid_support]["support_paths"]
        support_t = self._read_and_preprocess_supports(support_paths)

        # 3. Load query image (pre-extracted frame) and target boxes
        frame_path = video_path_query.parent / "frames" / f"frame_{fr:06d}.jpg"
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            raise FileNotFoundError(
                f"Missing pre-extracted frame: {frame_path}\nRun extract_frames.py first."
            )
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img, scale = resize_keep_aspect(img, self.query_short, self.query_long_cap)
        img_t = to_tensor_and_normalize(img)  # CPU

        boxes = np.array(self.ann[vid_query].get(fr, []), dtype=np.float32)
        if boxes.size > 0:
            boxes = scale_boxes(boxes, scale)

        if make_negative_episode:
            labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)  # all 0
        else:
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)   # all 1

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": labels,
            "image_id": torch.tensor(idx, dtype=torch.int64),
        }
        return support_t, img_t, target




# --------------------------------------------
# Factory
# --------------------------------------------
def make_loaders(
        data_root: Path = DATA_ROOT,
        annot_path: Path = ANNOT_PATH,
        samples_root: Path = SAMPLES_ROOT,
        novel_labels: Optional[List[str]] = None,
        batch_size_train: int = 4,
        num_workers: int = 4,
):
    ann_by_video = load_annotations(annot_path)
    sample_info = discover_samples(samples_root)
    splits = build_splits(ann_by_video, sample_info, novel_labels=novel_labels)

    # --- Step 4 Dataset ---
    train_ds = DetectionTrainDataset(splits["train_samples"], sample_info)

    # --- Step 7 and Validation Datasets ---
    # --- MODIFIED: Pass 'train_mode=True' and 'sample_info' ---
    train_ep_ds = EpisodeDataset(
        episodes=splits["train_episodes"],
        ann_by_video=ann_by_video,
        sample_info=sample_info,
        train_mode=True,
        negative_prob=0.0,  # or tune [0.5, 0.8]
    )
    val_ep_ds = EpisodeDataset(
        episodes=splits["val_episodes"],
        ann_by_video=ann_by_video,
        sample_info=sample_info,
        train_mode=False  # <-- Val/Test is always positive
    )
    test_ep_ds = EpisodeDataset(
        episodes=splits["test_episodes"],
        ann_by_video=ann_by_video,
        sample_info=sample_info,
        train_mode=False
    )
    # --- END MODIFIED ---

    # --- Step 4 Loader ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,  # Used by train_frcnn.py
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_detection,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # --- Step 7 Train Loader (NEW) ---
    train_episode_loader = DataLoader(
        train_ep_ds,
        batch_size=batch_size_train,  # Used by siamese.py (MUST be 1)
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        # No collate_fn, use default.
    )

    # --- Validation / Test Loaders ---
    val_ep_loader = DataLoader(
        val_ep_ds,
        batch_size=1,  # Always 1 for eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_ep_loader = DataLoader(
        test_ep_ds,
        batch_size=1,  # Always 1 for eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return {
        "train_loader": train_loader,  # For Step 4
        "train_episode_loader": train_episode_loader,  # For Step 7
        "val_episode_loader": val_ep_loader,
        "test_episode_loader": test_ep_loader,
        "splits": splits,
        "sample_info": sample_info,
        "ann_by_video": ann_by_video,
    }


# --------------------------------------------
# Self-check
# --------------------------------------------
if __name__ == "__main__":
    loaders = make_loaders(
        data_root=DATA_ROOT,
        annot_path=ANNOT_PATH,
        samples_root=SAMPLES_ROOT,
        novel_labels=None,
        batch_size_train=4,
        num_workers=2,
    )
    print("Base labels:", loaders["splits"]["base_labels"])
    print("Novel labels:", loaders["splits"]["novel_labels"])
    print("#train samples (for Step 4):", len(loaders["splits"]["train_samples"]))
    print("#train episodes (for Step 7):", len(loaders["splits"]["train_episodes"]))  # <-- NEW
    print("#val episodes:", len(loaders["splits"]["val_episodes"]))
    print("#test episodes:", len(loaders["splits"]["test_episodes"]))

    print("\nChecking train_loader (Step 4)...")
    img_b, tgt_b = next(iter(loaders["train_loader"]))
    print(f"  Batch images: {len(img_b)}, first shape: {img_b[0].shape}")
    print(f"  Batch targets: {len(tgt_b)}, first keys: {tgt_b[0].keys()}")

    print("\nChecking train_episode_loader (Step 7)...")
    # Note: this loader has batch_size=4 from main, but will be 1 for siamese.py
    s_b, q_b, t_b = next(iter(loaders["train_episode_loader"]))
    print(f"  Support batch shape: {s_b.shape}")
    print(f"  Query batch shape: {q_b.shape}")
    print(f"  Target batch length: {len(t_b)}")