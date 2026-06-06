from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_GROUPS = {
    "train": ("M1", "M2", "R1", "R2", "R3", "R4", "R5"),
    "test": ("M3", "R6"),
    "generalization": ("P1", "P2", "P3"),
}


@dataclass(frozen=True)
class PatchSpec:
    even_path: Path
    odd_path: Path
    group: str
    frame_index: int
    y: int
    x: int
    rotation: int


def _parse_frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _grid_starts(image_size: int, patch_size: int, stride: int) -> list[int]:
    if patch_size > image_size:
        raise ValueError(f"patch_size={patch_size} exceeds image_size={image_size}")
    starts = list(range(0, image_size - patch_size + 1, stride))
    last = image_size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _list_pairs(root: Path, group: str, sample_interval: int) -> list[tuple[Path, Path, int]]:
    even_dir = root / group / "even"
    odd_dir = root / group / "odd"
    if not even_dir.is_dir() or not odd_dir.is_dir():
        raise FileNotFoundError(f"Expected even/odd folders under {root / group}")

    even_files = sorted(even_dir.glob("*.npy"), key=_parse_frame_index)
    odd_by_index = {_parse_frame_index(path): path for path in odd_dir.glob("*.npy")}
    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")

    pairs = []
    for even_path in even_files[::sample_interval]:
        frame_index = _parse_frame_index(even_path)
        odd_path = odd_by_index.get(frame_index)
        if odd_path is None:
            raise FileNotFoundError(f"Missing odd pair for {even_path.name}")
        pairs.append((even_path, odd_path, frame_index))
    return pairs


def complex_to_channels(arr: np.ndarray, scale: float) -> torch.Tensor:
    channels = np.stack((arr.real, arr.imag), axis=0).astype(np.float32, copy=False)
    channels = np.ascontiguousarray(channels / scale)
    return torch.from_numpy(channels)


def channels_to_complex(channels: np.ndarray) -> np.ndarray:
    if channels.shape[0] != 2:
        raise ValueError(f"Expected 2 channels, got shape {channels.shape}")
    return channels[0].astype(np.float32) + 1j * channels[1].astype(np.float32)


class ComplexSvdN2NDataset(Dataset):
    """Noise2Noise dataset for paired complex SVD frames.

    Each sample is a fixed even->odd pair, represented as real/imag channels.
    The input patch magnitude defines the shared normalization scale.
    """

    def __init__(
        self,
        root: str | Path,
        groups: Iterable[str],
        patch_size: int = 256,
        stride: int = 128,
        sample_interval: int = 10,
        augment_rotations: bool = False,
        image_size: int = 512,
        scale_percentile: float = 99.9,
        eps: float = 1e-8,
    ) -> None:
        self.root = Path(root)
        self.groups = tuple(groups)
        self.patch_size = patch_size
        self.stride = stride
        self.sample_interval = sample_interval
        self.augment_rotations = augment_rotations
        self.image_size = image_size
        self.scale_percentile = scale_percentile
        self.eps = eps

        starts = _grid_starts(image_size, patch_size, stride)
        rotations = (0, 1, 2, 3) if augment_rotations else (0,)
        items: list[PatchSpec] = []
        self._frame_cache: dict[Path, np.ndarray] = {}
        for group in self.groups:
            for even_path, odd_path, frame_index in _list_pairs(self.root, group, sample_interval):
                for path in (even_path, odd_path):
                    if path not in self._frame_cache:
                        arr = np.load(path)
                        if arr.shape != (self.image_size, self.image_size):
                            raise ValueError(f"{path} has shape {arr.shape}, expected {(self.image_size, self.image_size)}")
                        if not np.iscomplexobj(arr):
                            raise TypeError(f"{path} is {arr.dtype}, expected a complex dtype")
                        self._frame_cache[path] = arr
                for y in starts:
                    for x in starts:
                        for rotation in rotations:
                            items.append(PatchSpec(even_path, odd_path, group, frame_index, y, x, rotation))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        spec = self.items[index]
        even = self._load_complex(spec.even_path)
        odd = self._load_complex(spec.odd_path)
        even_patch = self._crop(even, spec.y, spec.x, spec.rotation)
        odd_patch = self._crop(odd, spec.y, spec.x, spec.rotation)

        scale = max(float(np.percentile(np.abs(even_patch), self.scale_percentile)), self.eps)
        return {
            "input": complex_to_channels(even_patch, scale),
            "target": complex_to_channels(odd_patch, scale),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "group": spec.group,
            "frame_index": spec.frame_index,
            "y": spec.y,
            "x": spec.x,
            "rotation": spec.rotation,
        }

    def _load_complex(self, path: Path) -> np.ndarray:
        return self._frame_cache[path]

    def _crop(self, arr: np.ndarray, y: int, x: int, rotation: int) -> np.ndarray:
        patch = arr[y : y + self.patch_size, x : x + self.patch_size]
        if rotation:
            patch = np.rot90(patch, k=rotation)
        return np.ascontiguousarray(patch)


def make_datasets(
    root: str | Path,
    patch_size: int = 256,
    stride: int = 128,
    sample_interval: int = 10,
    image_size: int = 512,
    scale_percentile: float = 99.9,
    eps: float = 1e-8,
) -> dict[str, ComplexSvdN2NDataset]:
    return {
        "train": ComplexSvdN2NDataset(
            root,
            SPLIT_GROUPS["train"],
            patch_size=patch_size,
            stride=stride,
            sample_interval=sample_interval,
            augment_rotations=True,
            image_size=image_size,
            scale_percentile=scale_percentile,
            eps=eps,
        ),
        "test": ComplexSvdN2NDataset(
            root,
            SPLIT_GROUPS["test"],
            patch_size=patch_size,
            stride=stride,
            sample_interval=sample_interval,
            augment_rotations=False,
            image_size=image_size,
            scale_percentile=scale_percentile,
            eps=eps,
        ),
        "generalization": ComplexSvdN2NDataset(
            root,
            SPLIT_GROUPS["generalization"],
            patch_size=patch_size,
            stride=stride,
            sample_interval=sample_interval,
            augment_rotations=False,
            image_size=image_size,
            scale_percentile=scale_percentile,
            eps=eps,
        ),
    }
