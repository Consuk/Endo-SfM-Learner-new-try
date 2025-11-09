# datasets/validation_folders.py
# Validation dataset that can optionally return GT depth aligned by list order for EndoVis/SCARED.

import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as T

from .sequence_folders import (
    IMG_H, IMG_W, MEAN, STD, _parse_split, _imread_rgb, _build_intrinsics
)


class ValidationSet(data.Dataset):
    """
    If with GT is requested by train.py (--with-gt), this dataset should be used.
    It yields (tgt_img, depth) pairs, where depth is drawn from splits/endovis/gt_depth.npz
    and aligned strictly by list order with val_files.txt.
    """
    def __init__(self, data_root: str, transform=None, dataset: str = "endovis", with_gt: bool = True):
        super().__init__()
        if dataset.lower() != "endovis":
            raise ValueError("ValidationSet here only supports dataset='endovis'.")

        self.data_root = data_root
        self.transform = transform  # we normalize ourselves to match train.py
        self.with_gt = with_gt

        # read val split
        self.split_root = "splits"
        self.split_name = "endovis"
        self.samples: List[Tuple[str, int, str]] = _parse_split(self.split_root, self.split_name, train=False)

        # optional GT
        self.gt_path = os.path.join(self.split_root, self.split_name, "gt_depth.npz")
        if self.with_gt:
            if not os.path.isfile(self.gt_path):
                raise FileNotFoundError(
                    "with_gt=True but GT file not found at "
                    f"{self.gt_path}. Please place your gt_depth.npz here."
                )
            # Accept either default np.savez (arr_0) or named key 'depths'
            npz = np.load(self.gt_path, allow_pickle=True)
            if "depths" in npz:
                self.gt_depths = npz["depths"]
            else:
                # Assume a single unnamed array (arr_0) is the sequence of depth maps
                self.gt_depths = npz[list(npz.files)[0]]
            if len(self.gt_depths) != len(self.samples):
                raise ValueError(
                    f"GT length ({len(self.gt_depths)}) does not match val list length ({len(self.samples)}). "
                    "Because your GT is aligned by *list order*, these must match exactly."
                )
        else:
            self.gt_depths = None

        # I/O helpers
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BILINEAR)
        self.normalize = T.Normalize(mean=MEAN, std=STD)

        # intrinsics (not returned when with_gt=True because train.py expects (img, depth) only)
        self.K = _build_intrinsics(IMG_W, IMG_H)

    def __len__(self) -> int:
        return len(self.samples)

    def _img_path(self, folder: str, frame_idx: int, side: str) -> str:
        fname = f"{frame_idx}.jpg"  # matches your lists
        return os.path.join(self.data_root, folder, "data", fname)

    def _load_img_tensor(self, path: str) -> torch.Tensor:
        img = _imread_rgb(path)
        img = self.resize(img)
        ten = self.to_tensor(img)
        ten = self.normalize(ten)
        return ten

    def __getitem__(self, index: int):
        folder, frame_idx, side = self.samples[index]
        img_path = self._img_path(folder, frame_idx, side)
        tgt_img = self._load_img_tensor(img_path)

        if self.with_gt:
            # depth map expected as a 2D array; we resize to IMG_HxIMG_W to match network output comparably
            depth_np = self.gt_depths[index].astype(np.float32)
            if depth_np.ndim == 3:
                # handle HxWx1
                depth_np = depth_np[..., 0]
            # resize with PIL for simplicity
            d_img = Image.fromarray(depth_np)
            d_img = d_img.resize((IMG_W, IMG_H), resample=Image.BILINEAR)
            depth = torch.from_numpy(np.array(d_img, dtype=np.float32))  # HxW
            return tgt_img, depth
        else:
            # If not using GT: mirror train signature (img, [], K, K_inv) – but train.py won't call this branch with with_gt=False.
            K = torch.from_numpy(self.K.copy())
            K_inv = torch.inverse(K)
            return tgt_img, [], K, K_inv
