from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from svd_denoising.data import channels_to_complex, complex_to_channels


def compute_power_doppler(frames: Iterable[np.ndarray]) -> np.ndarray:
    pdi = None
    for frame in frames:
        power = np.square(np.abs(frame), dtype=np.float32)
        if pdi is None:
            pdi = power
        else:
            pdi += power
    if pdi is None:
        raise ValueError("At least one frame is required to compute PDI")
    return pdi.astype(np.float32, copy=False)


def pdi_to_db(pdi: np.ndarray, dynamic_range_db: float = 50.0, eps: float = 1e-12) -> np.ndarray:
    peak = max(float(np.max(pdi)), eps)
    pdi_db = 10.0 * np.log10(np.maximum(pdi, eps) / peak)
    return np.clip(pdi_db, -dynamic_range_db, 0.0).astype(np.float32)


def save_pdi_image(pdi: np.ndarray, save_path: Path, title: str, dynamic_range_db: float = 50.0) -> None:
    display = pdi_to_db(pdi, dynamic_range_db=dynamic_range_db)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(display, cmap="hot", vmin=-dynamic_range_db, vmax=0.0)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="dB")
    # Fix tick step so tight_layout behaves consistently across dynamic_range_db values
    if int(dynamic_range_db) % 5 == 0:
        step = 5
    elif int(dynamic_range_db) % 3 == 0:
        step = 3
    elif int(dynamic_range_db) % 2 == 0:
        step = 2
    else:
        step = 1
    cbar.set_ticks(list(range(0, -int(dynamic_range_db) - 1, -step)))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def estimate_background_noise(pdi: np.ndarray) -> np.ndarray:
    """Estimate 2D background noise by smoothing row/column profiles.

    Adapted from LinearSVD_v1 Noise_Estimation.
    """
    h, w = pdi.shape

    # Step 1: row profile (mean along columns), smooth, broadcast to 2D
    profile = np.mean(pdi, axis=1)
    window_length = max(1, w // 6)
    weights = np.ones(window_length, dtype=profile.dtype)
    num = np.convolve(profile, weights, mode="same")
    denom = np.convolve(np.ones_like(profile), weights, mode="same")
    denom = np.maximum(denom, 1e-12)
    smoothed = num / denom
    noise = np.repeat(smoothed.reshape(-1, 1), repeats=w, axis=1)

    # Step 2: column profile (mean along rows), smooth, multiply
    profile = np.mean(pdi, axis=0)
    window_length = max(1, h // 6)
    weights = np.ones(window_length, dtype=profile.dtype)
    num = np.convolve(profile, weights, mode="same")
    denom = np.convolve(np.ones_like(profile), weights, mode="same")
    denom = np.maximum(denom, 1e-12)
    smoothed = num / denom
    noise = noise * smoothed.reshape(1, -1)

    return noise


def _parse_frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


@torch.no_grad()
def denoise_complex_frames(
    model: torch.nn.Module,
    frames: Sequence[np.ndarray],
    device: torch.device,
    scale_percentile: float,
    eps: float = 1e-8,
) -> list[np.ndarray]:
    scales = [max(float(np.percentile(np.abs(frame), scale_percentile)), eps) for frame in frames]
    batch = torch.stack([complex_to_channels(frame, scale) for frame, scale in zip(frames, scales)]).to(device)
    outputs = model(batch).cpu().numpy()
    return [channels_to_complex(outputs[index]) * scales[index] for index in range(outputs.shape[0])]


@torch.no_grad()
def export_group_network_pdi(
    model: torch.nn.Module,
    full_root: str | Path,
    group: str,
    output_dir: Path,
    device: torch.device,
    scale_percentile: float,
    batch_size: int,
    dynamic_range_db: float = 50.0,
) -> dict[str, str | int]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_dir = Path(full_root) / group
    full_files = sorted(full_dir.glob("*.npy"), key=_parse_frame_index)
    if not full_files:
        raise FileNotFoundError(f"No full SVD frames found under {full_dir}")

    pdi = None

    for start in range(0, len(full_files), max(1, batch_size)):
        batch_files = full_files[start : start + max(1, batch_size)]
        frames = [np.load(path) for path in batch_files]
        outputs = denoise_complex_frames(model, frames, device, scale_percentile)
        batch_pdi = compute_power_doppler(outputs)
        if pdi is None:
            pdi = batch_pdi
        else:
            pdi += batch_pdi

    pdi_path = output_dir / f"{group}_network_pdi.npy"
    image_path = output_dir / f"{group}_network_pdi_{int(dynamic_range_db)}db.png"
    np.save(pdi_path, pdi.astype(np.float32, copy=False))
    save_pdi_image(pdi, image_path, title=f"{group} network PDI", dynamic_range_db=dynamic_range_db)
    return {
        "group": group,
        "frame_count": len(full_files),
        "pdi": pdi_path.name,
        "image": image_path.name,
    }


def export_group_noisy_pdi(
    full_root: str | Path,
    group: str,
    output_dir: Path,
    dynamic_range_db: float = 30.0,
) -> dict[str, str | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_dir = Path(full_root) / group
    full_files = sorted(full_dir.glob("*.npy"), key=_parse_frame_index)
    if not full_files:
        raise FileNotFoundError(f"No full SVD frames found under {full_dir}")

    frames = [np.load(path) for path in full_files]
    pdi = compute_power_doppler(frames)
    noise = estimate_background_noise(pdi)
    pdi_equalized = pdi / noise

    pdi_path = output_dir / f"{group}_noisy_svd_pdi.npy"
    image_path = output_dir / f"{group}_noisy_svd_pdi_{int(dynamic_range_db)}db.png"
    np.save(pdi_path, pdi_equalized.astype(np.float32, copy=False))
    save_pdi_image(pdi_equalized, image_path, title=f"{group} noisy SVD PDI", dynamic_range_db=dynamic_range_db)
    return {
        "group": group,
        "frame_count": len(full_files),
        "pdi": pdi_path.name,
        "image": image_path.name,
    }
