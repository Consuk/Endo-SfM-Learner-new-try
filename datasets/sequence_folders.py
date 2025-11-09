# datasets/sequence_folders.py
# Minimal sequence loader for EndoVis/SCARED compatible with train.py expectations.

import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as T


# === Config ===
IMG_H, IMG_W = 256, 320  # network input size (HxW). Change if your model expects a different size.

# SCARED normalized intrinsics (from your provided scared_dataset.py)
# These are "normalized" (0..1 over width/height). We scale them to pixel units below.
FX_NORM, FY_NORM = 0.5389, 0.6736
CX_NORM, CY_NORM = 0.5023, 0.4993

MEAN = [0.45, 0.45, 0.45]
STD  = [0.225, 0.225, 0.225]


def _build_intrinsics(w: int, h: int) -> np.ndarray:
    """Build 3x3 K (and leave 4x4 for callers if needed)."""
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = FX_NORM * w
    K[1, 1] = FY_NORM * h
    K[0, 2] = CX_NORM * w
    K[1, 2] = CY_NORM * h
    return K


def _parse_split(split_root: str, split_name: str, train: bool) -> List[Tuple[str, int, str]]:
    """
    Reads {train,val}_files.txt with lines like: 'dataset3/keyframe4\t390\tl'
    Returns list of (folder, frame_idx, side).
    """
    fname = "train_files.txt" if train else "val_files.txt"
    fpath = os.path.join(split_root, split_name, fname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Split list not found: {fpath}")

    entries = []
    with open(fpath, "r") as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 2:
                continue
            folder = s[0]
            frame_idx = int(s[1])
            side = s[2] if len(s) > 2 else "l"
            entries.append((folder, frame_idx, side))
    if len(entries) == 0:
        raise RuntimeError(f"No entries found in split list {fpath}")
    return entries


def _imread_rgb(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


class SequenceFolder(data.Dataset):
    """
    EndoVis/SCARED sequence dataset that returns (tgt_img, ref_imgs, K, K_inv)
    exactly as train.py expects.

    Directory layout expected under data root:
      <data_root>/
        datasetX/keyframeY/data/<frame>.jpg

    Split files under:
      splits/endovis/{train,val}_files.txt  with lines: "datasetX/keyframeY <frame> l"
    """
    def __init__(
        self,
        data_root: str,
        transform=None,
        seed: int = 0,
        train: bool = True,
        sequence_length: int = 3,
        dataset: str = "endovis",
        img_ext: str = ".jpg",
    ) -> None:
        super().__init__()
        if dataset.lower() != "endovis":
            raise ValueError(
                f"This SequenceFolder only supports dataset='endovis' for SCARED. Got: {dataset}"
            )

        self.data_root = data_root
        self.train = train
        self.seq_len = sequence_length
        if self.seq_len < 1 or self.seq_len % 2 == 0:
            # typical is odd (e.g., 3 -> offsets [-1,0,1])
            raise ValueError(f"sequence_length must be odd and >=1, got {self.seq_len}")

        # neighbors to use (e.g., [-1, +1] for seq_len=3)
        half = self.seq_len // 2
        self.frame_offsets = [off for off in range(-half, half + 1) if off != 0]

        self.img_ext = img_ext
        self.transform = transform  # we will normalize ourselves to match train.py

        # read split entries
        self.split_root = "splits"
        self.split_name = "endovis"
        self.samples = _parse_split(self.split_root, self.split_name, train=self.train)

        # "Scenes" count is the number of unique folders
        self.scenes = sorted({folder for (folder, _, _) in self.samples})

        # image normalizer
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BILINEAR)
        self.normalize = T.Normalize(mean=MEAN, std=STD)

        # intrinsics at network resolution
        self.K = _build_intrinsics(IMG_W, IMG_H)
        self.K_inv = np.linalg.inv(self.K).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _img_path(self, folder: str, frame_idx: int, side: str) -> str:
        # SCARED: data under .../data/<frame>.<ext> ; side is always 'l' in your lists
        fname = f"{frame_idx}{self.img_ext}"
        return os.path.join(self.data_root, folder, "data", fname)

    def _load_img_tensor(self, path: str) -> torch.Tensor:
        img = _imread_rgb(path)
        img = self.resize(img)
        ten = self.to_tensor(img)
        ten = self.normalize(ten)
        return ten  # CxHxW

    def __getitem__(self, index: int):
        folder, frame_idx, side = self.samples[index]
        # target
        tgt_path = self._img_path(folder, frame_idx, side)
        tgt_img = self._load_img_tensor(tgt_path)

        # references (neighbors in same sequence if files exist)
        ref_imgs = []
        for off in self.frame_offsets:
            nb_idx = frame_idx + off
            nb_path = self._img_path(folder, nb_idx, side)
            if os.path.isfile(nb_path):
                ref_imgs.append(self._load_img_tensor(nb_path))
            # else: neighbor missing at boundary, safely skip

        intrinsics = torch.from_numpy(self.K.copy())      # 3x3
        intrinsics_inv = torch.from_numpy(self.K_inv.copy())  # 3x3

        return tgt_img, ref_imgs, intrinsics, intrinsics_inv
