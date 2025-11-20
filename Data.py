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
# Repo layout:
#   DATA_ROOT/
#     annotations/annotations.json
#     samples/<ClassN>_<idx>/{drone_video.mp4, object_images/img_1.jpg..img_3.jpg}

DATA_ROOT = Path("/home/chucky/Downloads/train/")  # change if needed
ANNOT_PATH = DATA_ROOT / "annotations" / "annotations.json"
SAMPLES_ROOT = DATA_ROOT / "samples"

# Normalization expected by torchvision detection models (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --------------------------------------------
# Utilities
# --------------------------------------------
def parse_label_and_index(video_id: str) -> Tuple[str, Optional[int]]:
    """'Backpack_0' -> ('Backpack', 0). If no suffix, returns (name, None)."""
    if "_" in video_id:
        name, idx = video_id.rsplit("_", 1)
        try:
            return name, int(idx)
        except ValueError:
            return name, None
    return video_id, None

def load_annotations(annot_path: Path) -> Dict[str, Dict[int, List[List[int]]]]:
    """
    Returns: dict[video_id] -> dict[frame_idx] -> list of boxes [x1,y1,x2,y2]
    Input JSON format (example given by user):
    [
      {
        "video_id": "Backpack_0",
        "annotations": [
          { "bboxes": [ {"frame": 3483, "x1":..,"y1":..,"x2":..,"y2":..}, ... ] },
          ...
        ]
      },
      ...
    ]
    """
    with open(annot_path, "r") as f:
        data = json.load(f)

    by_video: Dict[str, Dict[int, List[List[int]]]] = {}
    for item in data:
        vid = item["video_id"]
        frames: Dict[int, List[List[int]]] = by_video.setdefault(vid, {})
        ann_list = item.get("annotations", [])
        for seg in ann_list:
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
    out = {}
    for d in sorted(samples_root.iterdir()):
        if not d.is_dir():
            continue
        video_id = d.name  # e.g., Backpack_0
        video_path = d / "drone_video.mp4"
        supp_dir = d / "object_images"
        supports = [supp_dir / f"img_{i}.jpg" for i in (1, 2, 3)]
        if not video_path.exists():  # skip if missing
            continue
        if not all(p.exists() for p in supports):
            continue
        cls, _ = parse_label_and_index(video_id)
        out[video_id] = {
            "video_path": video_path,
            "support_paths": supports,
            "class_name": cls,
        }
    return out

def cv2_read_frame_at(video_path: Path, frame_idx: int) -> np.ndarray:
    """
    Read a specific frame by index (0-based). Returns RGB uint8 HxWx3.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    # Some codecs are picky; setting frame position can be slow but OK for small datasets.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

def to_tensor_and_normalize(img_rgb: np.ndarray) -> torch.Tensor:
    """
    HxWx3 uint8 -> FloatTensor CxHxW in [0,1], normalized by ImageNet stats.
    """
    img = torch.from_numpy(img_rgb).float() / 255.0  # H W C
    img = img.permute(2, 0, 1)  # C H W
    img = F.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return img

def resize_keep_aspect(img: np.ndarray, short_side: int = 800, long_cap: int = 1333) -> Tuple[np.ndarray, float]:
    """
    Resize image so that min(H,W) == short_side (unless already smaller) and max(H,W) <= long_cap.
    Returns resized image and scale factor applied to both axes (uniform scaling).
    """
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if max(h, w) * scale > long_cap:
        scale = long_cap / max(h, w)
    if scale != 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img, scale

def scale_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    return boxes * scale

def collate_fn_detection(batch):
    # For torchvision detection models
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

# --------------------------------------------
# Split builder
# --------------------------------------------
def build_splits(
    ann_by_video: Dict[str, Dict[int, List[List[int]]]],
    sample_info: Dict[str, Dict[str, Any]],
    novel_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Returns:
      {
        "train_samples": List[ (video_id, frame_idx, boxes) ],
        "val_episodes":   List[ (video_id, frame_idx) ],
        "test_episodes":  List[ (video_id, frame_idx) ],
        "base_labels":    List[str],
        "novel_labels":   List[str]
      }
    """
    # Discover all class names present
    by_class: Dict[str, List[str]] = {}
    for vid in ann_by_video.keys():
        cls, _ = parse_label_and_index(vid)
        by_class.setdefault(cls, []).append(vid)

    classes = sorted(by_class.keys())

    if novel_labels is None:
        # Default: last 2 classes alphabetically as novel (deterministic fallback)
        novel_labels = classes[-2:] if len(classes) >= 2 else []
    base_labels = [c for c in classes if c not in set(novel_labels)]

    # Helper mapping for quick access
    def is_base(vid: str) -> bool:
        cls, _ = parse_label_and_index(vid)
        return cls in base_labels

    def is_novel(vid: str) -> bool:
        cls, _ = parse_label_and_index(vid)
        return cls in novel_labels

    train_samples = []
    val_episodes = []
    test_episodes = []

    for vid, frames_dict in ann_by_video.items():
        # Skip videos missing assets
        if vid not in sample_info:
            continue

        # Decide where this video goes
        cls, idx = parse_label_and_index(vid)
        frame_ids = sorted(frames_dict.keys())

        if is_base(vid):
            if idx is not None:
                # convention: _0 -> train, _1 -> val
                if idx == 0:
                    for fr in frame_ids:
                        boxes = np.array(frames_dict[fr], dtype=np.float32)  # [N,4] x1y1x2y2
                        train_samples.append((vid, fr, boxes))
                else:
                    for fr in frame_ids:
                        val_episodes.append((vid, fr))
            else:
                # no index suffix -> do 80/20 frame split
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

    return {
        "train_samples": train_samples,
        "val_episodes": val_episodes,
        "test_episodes": test_episodes,
        "base_labels": base_labels,
        "novel_labels": novel_labels,
    }

# --------------------------------------------
# Datasets
# --------------------------------------------
class DetectionTrainDataset(Dataset):
    """Class-agnostic detection dataset for torchvision models.

    Optionally caches all training frames in RAM so we don't keep
    re-opening the same videos every epoch.
    """

    def __init__(
            self,
            train_samples: List[Tuple[str, int, np.ndarray]],
            sample_info: Dict[str, Dict[str, Any]],
            resize_short: int = 800,
            resize_long_cap: int = 1333,
            hflip_p: float = 0.5,
            cache_in_ram: bool = False,      # <--- NEW
    ):
        self.samples = train_samples
        self.info = sample_info
        self.resize_short = resize_short
        self.resize_long_cap = resize_long_cap
        self.hflip_p = hflip_p
        self.rng = np.random.RandomState(1337)

        self.cache_in_ram = cache_in_ram
        self.frame_cache: Dict[Tuple[str, int], np.ndarray] = {}

        # ---- Optional preload: read all unique (video, frame) once ----
        if self.cache_in_ram:
            unique_keys = {(vid, fr) for (vid, fr, _) in self.samples}
            print(f"[DetectionTrainDataset] Preloading {len(unique_keys)} frames into RAM...")
            for vid, fr in sorted(unique_keys):
                video_path = self.info[vid]["video_path"]
                img = cv2_read_frame_at(video_path, fr)  # RGB uint8
                self.frame_cache[(vid, fr)] = img
            print(f"[DetectionTrainDataset] Cached {len(self.frame_cache)} frames in RAM.")

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

    def _get_frame(self, vid: str, fr: int) -> np.ndarray:
        """Return RGB frame either from RAM cache or by reading from disk."""
        key = (vid, fr)
        if self.cache_in_ram:
            img = self.frame_cache.get(key, None)
            if img is None:
                # Fallback: if not preloaded, read once and cache
                video_path = self.info[vid]["video_path"]
                img = cv2_read_frame_at(video_path, fr)
                self.frame_cache[key] = img
            # IMPORTANT: copy so that augmentations don't modify cached array
            return img.copy()
        else:
            video_path = self.info[vid]["video_path"]
            return cv2_read_frame_at(video_path, fr)

    def __getitem__(self, idx: int):
        vid, fr, boxes = self.samples[idx]

        # ---- Get frame from RAM instead of disk every time ----
        img = self._get_frame(vid, fr)   # RGB uint8

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
    Episodic dataset for validation/test:
      Each item = (support_tensors: (3,C,Hs,Ws), query_tensor: (C,Hq,Wq), target dict)
      Supports come from object_images/img_1..3.jpg in the SAME video folder.
    """
    def __init__(
        self,
        episodes: List[Tuple[str, int]],
        ann_by_video: Dict[str, Dict[int, List[List[int]]]],
        sample_info: Dict[str, Dict[str, Any]],
        support_size: int = 256,
        query_short: int = 800,
        query_long_cap: int = 1333,
    ):
        self.episodes = episodes
        self.ann = ann_by_video
        self.info = sample_info
        self.support_size = support_size
        self.query_short = query_short
        self.query_long_cap = query_long_cap

    def __len__(self):
        return len(self.episodes)

    def _read_and_preprocess_supports(self, support_paths: List[Path]) -> torch.Tensor:
        supp_tensors = []
        for p in support_paths:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                raise RuntimeError(f"Cannot read support image: {p}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.support_size, self.support_size), interpolation=cv2.INTER_LINEAR)
            supp_tensors.append(to_tensor_and_normalize(img))
        return torch.stack(supp_tensors, dim=0)  # (3, C, H, W)

    def __getitem__(self, idx: int):
        vid, fr = self.episodes[idx]
        vinfo = self.info[vid]
        video_path = vinfo["video_path"]  # Keep this line, we use it for the path
        support_paths = vinfo["support_paths"]

        support_t = self._read_and_preprocess_supports(support_paths)  # CPU

        # --- MODIFIED ---
        frame_path = vinfo["video_path"].parent / "frames" / f"frame_{fr:06d}.jpg"
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Missing pre-extracted frame: {frame_path}\nRun extract_frames.py first.")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # RGB
        # --- END MODIFIED ---

        img, scale = resize_keep_aspect(img, self.query_short, self.query_long_cap)
        img_t = to_tensor_and_normalize(img)

        # Targets for this frame
        boxes = np.array(self.ann[vid].get(fr, []), dtype=np.float32)
        if boxes.size > 0:
            boxes = scale_boxes(boxes, scale)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return support_t, img_t, target

# --------------------------------------------
# Factory functions
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

    train_ds = DetectionTrainDataset(splits["train_samples"], sample_info)
    val_ep_ds = EpisodeDataset(splits["val_episodes"], ann_by_video, sample_info)
    test_ep_ds = EpisodeDataset(splits["test_episodes"], ann_by_video, sample_info)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn_detection, pin_memory=True
    )
    val_ep_loader = DataLoader(
        val_ep_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_ep_loader = DataLoader(
        test_ep_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return {
        "train_loader": train_loader,
        "val_episode_loader": val_ep_loader,
        "test_episode_loader": test_ep_loader,
        "splits": splits,
        "sample_info": sample_info,
        "ann_by_video": ann_by_video,
    }

# --------------------------------------------
# Example usage
# --------------------------------------------
if __name__ == "__main__":
    # You SHOULD set your novel labels explicitly (example below).
    # With your classes: Backpack, Jacket, Laptop, Lifering, MobilePhone, Person1, WaterBottle
    # If you plan to keep 3 novel labels, e.g.:
    # novel = ["Lifering", "MobilePhone", "WaterBottle"]
    novel = ["Lifering", "MobilePhone", "WaterBottle"]

    loaders = make_loaders(
        data_root=DATA_ROOT,
        annot_path=ANNOT_PATH,
        samples_root=SAMPLES_ROOT,
        novel_labels=novel,
        batch_size_train=4,
        num_workers=4,
    )

    print("Base labels:", loaders["splits"]["base_labels"])
    print("Novel labels:", loaders["splits"]["novel_labels"])
    print("#train samples:", len(loaders["splits"]["train_samples"]))
    print("#val episodes:", len(loaders["splits"]["val_episodes"]))
    print("#test episodes:", len(loaders["splits"]["test_episodes"]))
