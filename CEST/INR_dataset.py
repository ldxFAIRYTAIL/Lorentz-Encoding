from dataclasses import dataclass
from typing import List, Literal

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetSplit:
    train_dataset: Dataset
    test_dataset: Dataset
    offsets: np.ndarray
    scale_factor: float
    affine: np.ndarray


class RandomPointsDataset(Dataset):
    def __init__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        offsets: List,
        indices: List,
        points_num: int,
        device: torch.device,
        seed: int = 42,
    ):
        super().__init__()
        self.device = device
        self.points_num = points_num
        if image.dtype != torch.float32:
            raise TypeError(f"Expected float32 image, got {image.dtype}")
        if image.shape[-2] != len(indices):
            raise ValueError(f"Channel count mismatch: {image.shape[-1]} vs {len(indices)}")
        if mask is not None and torch.unique(mask).numel() != 2:
            raise ValueError(f"Mask must be binary, got values {torch.unique(mask)}")

        self.image = image.to(self.device)
        self.mask = mask.to(self.device) if mask is not None else None
        if self.mask is not None and self.image.shape != self.mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {self.image.shape} vs {self.mask.shape}")

        self.offsets = torch.tensor(offsets, device=self.device, dtype=torch.float32)
        self.indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        self.dim_sizes = self.image.shape[:-1]
        self.image_size = np.prod(self.dim_sizes)
        self.coord_size = len(self.image.shape[:-1])
        self.value_size = self.image.shape[-1]
        self.point_indices, self.point_coords_norm, self.point_values, self.mask_values = self._get_point_coords()
        self.R = np.random.RandomState(seed)

    def _get_point_coords(self):
        point_indices = np.meshgrid(*[np.arange(i) for i in self.dim_sizes], indexing="ij")
        point_indices = np.stack(point_indices, axis=-1).reshape(-1, len(self.dim_sizes))
        point_indices = torch.tensor(point_indices, device=self.device, dtype=torch.long)

        point_values = self.image[tuple(point_indices.T)].type(torch.float32)
        mask_values = self.mask[tuple(point_indices.T)].type(torch.float32) if self.mask is not None else None

        spatial_dims = torch.tensor(self.dim_sizes, device=self.device, dtype=torch.float32)
        point_coords_norm = (point_indices / (spatial_dims / 2) - 1).type(torch.float32)

        z_indices = point_indices[:, -1].long()
        offset_values = self.offsets[self.indices[z_indices]]
        point_coords_norm[:, -1] = offset_values
        mask_indices = torch.nonzero(mask_values.squeeze()).squeeze()
        point_indices, point_coords_norm, point_values, mask_values = [
            x[mask_indices]
            for x in [point_indices, point_coords_norm, point_values, mask_values]
        ]
        if mask_values is not None and not torch.all(mask_values.squeeze() > 0):
            raise ValueError("Mask filtering failed.")
        return point_indices, point_coords_norm, point_values, mask_values

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        random_indices = self.R.randint(0, self.point_coords_norm.shape[0], (self.points_num,))
        random_indices = torch.tensor(random_indices, device=self.device, dtype=torch.int64)
        random_point_indices = self.point_indices[random_indices]
        random_point_coords = self.point_coords_norm[random_indices]
        random_point_values = self.point_values[random_indices]
        random_mask_values = self.mask_values[random_indices] if self.mask is not None else None
        return random_point_indices, random_point_coords, random_point_values, random_mask_values


def _select_volume_slices(cest_nii, img_path: str):
    data = cest_nii.get_fdata()
    if data.ndim == 3:
        return data
    if "phantom" in str(img_path):
        return data[..., 1:]
    if "B0corr" in str(img_path):
        return data
    return data[..., 2:]


def _apply_mask(cest_imgs, mask_path):
    if mask_path is None:
        return cest_imgs, None
    mask = nib.load(mask_path).get_fdata()
    try:
        cest_imgs[mask == 0] = 0
    except ValueError:
        cest_imgs = cest_imgs * mask
    return cest_imgs, mask


def _train_ppms(split_method: str, offset_path: str):
    if str(split_method).endswith("1"):
        return [
            -50, -10, -8, -4.5, -4, -3.5, -3, -2.5, -1, -0.5, 0,
            0.5, 1, 2.5, 3, 3.5, 4, 4.5, 8, 10, 50,
        ]
    if str(split_method).endswith("2"):
        return [
            -50, -20, -10, -8, -6, -5, -4.5, -4, -3.5, -3, -2.5, -1.5, -1, -0.5, 0,
            0.5, 1, 1.5, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10, 20, 50,
        ]
    if str(split_method).endswith("3"):
        return [
            -50, -35, -25, -20, -14, -10, -9, -8, -7, -6, -5, -4.5, -4, -3.5, -3,
            -2.5, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8,
            9, 10, 14, 20, 25, 35, 50,
        ]
    if str(split_method).startswith("normal"):
        return pd.read_csv(str(offset_path).replace("offsets_dense", "offsets"))["offset"].values[2:]
    raise ValueError(f"Unknown split method: {split_method}")


def get_train_test_dataset(
    img_path: str,
    offset_path: str,
    mask_path: str = None,
    split_method: Literal["pattern1", "pattern2", "pattern3"] = "pattern2",
    slice: bool = False,
    points_per_sample: int = 20480,
    slice_index: int = None,
    device: torch.device = "cpu",
) -> DatasetSplit:
    cest_nii = nib.load(img_path)
    cest_imgs = _select_volume_slices(cest_nii, img_path)
    cest_imgs, mask = _apply_mask(cest_imgs, mask_path)

    offsets_df = pd.read_csv(offset_path)
    offsets = offsets_df["offset"].values[2:]

    cest_imgs = np.clip(cest_imgs, a_min=0, a_max=1)
    train_ppms = _train_ppms(split_method, offset_path)
    train_ids = sorted(list(offsets).index(ppm) for ppm in train_ppms)
    test_ids = [
        idx
        for idx in range(cest_imgs.shape[-1])
        if idx not in train_ids and min(train_ids) < idx <= max(train_ids)
    ]

    if slice:
        train_gt_imgs = [cest_imgs[:, slice_index, :, i] for i in train_ids]
        test_gt_imgs = [cest_imgs[:, slice_index, :, i] for i in test_ids]
    else:
        train_gt_imgs = [cest_imgs[..., i] for i in train_ids]
        test_gt_imgs = [cest_imgs[..., i] for i in test_ids]

    train_gt_imgs = torch.tensor(np.stack(train_gt_imgs, axis=-1), dtype=torch.float32).unsqueeze(dim=-1)
    test_gt_imgs = torch.tensor(np.stack(test_gt_imgs, axis=-1), dtype=torch.float32).unsqueeze(dim=-1)

    mask_ = torch.tensor(mask, dtype=torch.float32) if mask is not None else None
    if mask_ is not None:
        mask_ = mask_.view(*mask.shape, 1, 1)
        if len(mask_.squeeze().shape) == 2:
            train_mask = mask_.repeat(1, 1, len(train_ids), 1)
            test_mask = mask_.repeat(1, 1, len(test_ids), 1)
        else:
            train_mask = mask_.repeat(1, 1, 1, len(train_ids), 1)
            test_mask = mask_.repeat(1, 1, 1, len(test_ids), 1)
        if slice:
            train_mask = train_mask[slice_index, ...]
            test_mask = test_mask[slice_index, ...]
    else:
        train_mask = None
        test_mask = None

    scale_factor = 50
    normalized_offsets = np.array([offset / scale_factor for offset in offsets])

    train_dataset = RandomPointsDataset(
        train_gt_imgs,
        train_mask,
        normalized_offsets,
        train_ids,
        points_num=points_per_sample,
        device=device,
    )
    test_dataset = RandomPointsDataset(
        test_gt_imgs,
        test_mask,
        normalized_offsets,
        test_ids,
        points_num=points_per_sample,
        device=device,
    )

    return DatasetSplit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        offsets=normalized_offsets,
        scale_factor=scale_factor,
        affine=cest_nii.affine,
    )
