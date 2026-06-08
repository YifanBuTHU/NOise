"""
Generate traditional denoising baselines for all datasets in N2N_result.
Saves baseline_gaussian.png, baseline_nlm.png, baseline_bm3d.png
into each sample directory alongside before_network.png / after_network.png / gt.png.

Dependencies:
    pip install opencv-python-headless bm3d

Usage:
    python generate_baselines.py
"""

import re
from pathlib import Path

import cv2
import numpy as np


N2N_RESULT_ROOT = Path(r"F:\NOise_data\N2N_result")
DATASETS = ["FMD", "US", "CT", "MRC", "CARE_2D"]

# Default parameters for each dataset (images are uint8 [0, 255])
DEFAULT_CONFIG = {
    "gaussian_sigma": 1.5,
    "nlm_h": 10.0,
    "bm3d_sigma_psd": 0.10,
}


def parse_ct_sigma_from_dirname(dirname: str) -> float | None:
    """Parse gaussian sigma from CT directory names like 'ct_000_gaussian_sigma003'."""
    m = re.search(r"gaussian_sigma(\d+)", dirname)
    if not m:
        return None
    digits = m.group(1)
    # e.g. sigma003 -> 0.03, sigma015 -> 0.15, sigma1 -> 0.1
    if len(digits) == 3:
        return int(digits) / 100.0
    return int(digits) / (10.0 ** len(digits))


def apply_gaussian(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Apply Gaussian blur."""
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_nlm(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    """Apply Non-local Means denoising via OpenCV."""
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def apply_bm3d(img: np.ndarray, sigma_psd: float = 0.10) -> np.ndarray:
    """Apply BM3D denoising. Input uint8 [0,255]; output uint8 [0,255]."""
    import bm3d

    img_float = img.astype(np.float32) / 255.0
    denoised = bm3d.bm3d(
        img_float,
        sigma_psd=sigma_psd,
        stage_arg=bm3d.BM3DStages.ALL_STAGES,
    )
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    return denoised


def process_sample(sample_dir: Path, config: dict) -> None:
    before_path = sample_dir / "before_network.png"
    if not before_path.exists():
        return

    img = cv2.imread(str(before_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [WARN] Could not read {before_path}")
        return

    # Gaussian
    gaussian = apply_gaussian(img, sigma=config["gaussian_sigma"])
    cv2.imwrite(str(sample_dir / "baseline_gaussian.png"), gaussian)

    # NLM
    nlm = apply_nlm(img, h=config["nlm_h"])
    cv2.imwrite(str(sample_dir / "baseline_nlm.png"), nlm)

    # BM3D
    bm3d_img = apply_bm3d(img, sigma_psd=config["bm3d_sigma_psd"])
    cv2.imwrite(str(sample_dir / "baseline_bm3d.png"), bm3d_img)

    print(f"  Done: {sample_dir.name}")


def process_dataset(dataset_name: str) -> None:
    samples_dir = (
        N2N_RESULT_ROOT / dataset_name / "n2n_raw_unetv2_256_e300" / "samples"
    )
    if not samples_dir.exists():
        print(f"[WARN] Samples dir not found: {samples_dir}")
        return

    sample_dirs = [d for d in sorted(samples_dir.iterdir()) if d.is_dir()]
    print(f"Processing {dataset_name}: {len(sample_dirs)} samples")

    for sample_dir in sample_dirs:
        config = DEFAULT_CONFIG.copy()

        # CT-specific: try to parse sigma from directory name
        if dataset_name == "CT":
            parsed_sigma = parse_ct_sigma_from_dirname(sample_dir.name)
            if parsed_sigma is not None:
                config["bm3d_sigma_psd"] = parsed_sigma
                # Heuristic: scale NLM h and Gaussian sigma with noise level
                config["nlm_h"] = max(3.0, parsed_sigma * 255.0 * 0.5)
                config["gaussian_sigma"] = max(0.5, parsed_sigma * 20.0)

        process_sample(sample_dir, config)

    print(f"Finished {dataset_name}\n")


if __name__ == "__main__":
    for ds in DATASETS:
        process_dataset(ds)
