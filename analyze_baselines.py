"""
Compute quantitative metrics for all baselines and N2N results.

Outputs:
    1. per_sample_metrics.csv   – one row per sample with all method metrics
    2. per_dataset_summary.csv  – aggregated mean ± std per dataset
    3. per_dataset_summary.md   – human-readable markdown tables

Metrics:
    - If gt.png exists: PSNR (dB) ↑ and SSIM ↑ vs GT
    - If gt.png missing: NoiseEst ↓ and Sharpness ↑ (proxy metrics)

Usage:
    python analyze_baselines.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


N2N_RESULT_ROOT = Path(r"F:\NOise_data\N2N_result")
DATASETS = ["FMD", "US", "CT", "MRC", "CARE_2D"]

METHODS = [
    ("before_network", "Noisy Input"),
    ("after_network", "N2N"),
    ("baseline_gaussian", "Gaussian"),
    ("baseline_nlm", "NLM"),
    ("baseline_bm3d", "BM3D"),
]


def load_gray(path: Path) -> np.ndarray:
    """Load image as grayscale float64 [0, 255]."""
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def compute_psnr(img: np.ndarray, ref: np.ndarray, data_range: float = 255.0) -> float:
    return float(peak_signal_noise_ratio(img, ref, data_range=data_range))


def compute_ssim(img: np.ndarray, ref: np.ndarray, data_range: float = 255.0) -> float:
    return float(structural_similarity(img, ref, data_range=data_range))


def estimate_noise_sigma(img_uint8: np.ndarray) -> float:
    """MAD estimator on Laplacian (proxy for residual noise). Lower = cleaner."""
    lap = cv2.Laplacian(img_uint8, cv2.CV_64F)
    sigma = float(np.median(np.abs(lap))) / 0.6745
    return sigma


def compute_sharpness(img_uint8: np.ndarray) -> float:
    """Tenengrad (mean squared Sobel gradient). Higher = sharper edges."""
    sx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1)
    return float(np.mean(sx**2 + sy**2))


def process_sample(sample_dir: Path, dataset_name: str) -> dict:
    gt_path = sample_dir / "gt.png"
    has_gt = gt_path.exists()
    gt = load_gray(gt_path) if has_gt else None

    row = {"dataset": dataset_name, "sample": sample_dir.name, "has_gt": has_gt}

    for key, label in METHODS:
        img_path = sample_dir / f"{key}.png"
        if not img_path.exists():
            row[f"{key}_psnr"] = None
            row[f"{key}_ssim"] = None
            row[f"{key}_noise_est"] = None
            row[f"{key}_sharpness"] = None
            continue

        img = load_gray(img_path)
        img_u8 = img.astype(np.uint8)

        if has_gt:
            row[f"{key}_psnr"] = compute_psnr(img, gt)
            row[f"{key}_ssim"] = compute_ssim(img, gt)
            row[f"{key}_noise_est"] = None
            row[f"{key}_sharpness"] = None
        else:
            row[f"{key}_psnr"] = None
            row[f"{key}_ssim"] = None
            row[f"{key}_noise_est"] = estimate_noise_sigma(img_u8)
            row[f"{key}_sharpness"] = compute_sharpness(img_u8)

    return row


def summarize(rows: list[dict], dataset_name: str) -> dict:
    """Aggregate mean ± std for a dataset."""
    summary = {"dataset": dataset_name, "n_samples": len(rows)}
    has_gt = rows[0]["has_gt"] if rows else False
    summary["has_gt"] = has_gt

    for key, label in METHODS:
        if has_gt:
            psnrs = [r[f"{key}_psnr"] for r in rows if r[f"{key}_psnr"] is not None]
            ssims = [r[f"{key}_ssim"] for r in rows if r[f"{key}_ssim"] is not None]
            summary[f"{label}_psnr_mean"] = float(np.mean(psnrs)) if psnrs else None
            summary[f"{label}_psnr_std"] = float(np.std(psnrs)) if psnrs else None
            summary[f"{label}_ssim_mean"] = float(np.mean(ssims)) if ssims else None
            summary[f"{label}_ssim_std"] = float(np.std(ssims)) if ssims else None
        else:
            noises = [r[f"{key}_noise_est"] for r in rows if r[f"{key}_noise_est"] is not None]
            sharps = [r[f"{key}_sharpness"] for r in rows if r[f"{key}_sharpness"] is not None]
            summary[f"{label}_noise_mean"] = float(np.mean(noises)) if noises else None
            summary[f"{label}_noise_std"] = float(np.std(noises)) if noises else None
            summary[f"{label}_sharp_mean"] = float(np.mean(sharps)) if sharps else None
            summary[f"{label}_sharp_std"] = float(np.std(sharps)) if sharps else None

    return summary


def write_per_sample_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summaries: list[dict], out_path: Path) -> None:
    if not summaries:
        return
    fieldnames = sorted(set().union(*(s.keys() for s in summaries)))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_summary_md(summaries: list[dict], out_path: Path) -> None:
    lines = ["# Baseline vs N2N Quantitative Results\n"]

    for s in summaries:
        ds = s["dataset"]
        n = s["n_samples"]
        has_gt = s["has_gt"]
        lines.append(f"## {ds} (n={n})\n")

        if has_gt:
            lines.append("| Method | PSNR (dB) ↑ | SSIM ↑ |")
            lines.append("|---|---|---|")
            for key, label in METHODS:
                pm = s.get(f"{label}_psnr_mean")
                ps = s.get(f"{label}_psnr_std")
                sm = s.get(f"{label}_ssim_mean")
                ss = s.get(f"{label}_ssim_std")
                psnr_str = f"{pm:.2f} ± {ps:.2f}" if pm is not None else "N/A"
                ssim_str = f"{sm:.4f} ± {ss:.4f}" if sm is not None else "N/A"
                lines.append(f"| {label} | {psnr_str} | {ssim_str} |")
        else:
            lines.append("| Method | NoiseEst ↓ | Sharpness ↑ |")
            lines.append("|---|---|---|")
            for key, label in METHODS:
                nm = s.get(f"{label}_noise_mean")
                ns = s.get(f"{label}_noise_std")
                sm = s.get(f"{label}_sharp_mean")
                ss = s.get(f"{label}_sharp_std")
                noise_str = f"{nm:.2f} ± {ns:.2f}" if nm is not None else "N/A"
                sharp_str = f"{sm:.1f} ± {ss:.1f}" if sm is not None else "N/A"
                lines.append(f"| {label} | {noise_str} | {sharp_str} |")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    all_rows = []
    summaries = []

    for ds in DATASETS:
        samples_dir = N2N_RESULT_ROOT / ds / "n2n_raw_unetv2_256_e300" / "samples"
        if not samples_dir.exists():
            print(f"[WARN] Not found: {samples_dir}")
            continue

        sample_dirs = [d for d in sorted(samples_dir.iterdir()) if d.is_dir()]
        print(f"Processing {ds}: {len(sample_dirs)} samples")

        rows = []
        for sd in sample_dirs:
            try:
                row = process_sample(sd, ds)
                rows.append(row)
            except Exception as e:
                print(f"  [ERR] {sd.name}: {e}")
        all_rows.extend(rows)
        summaries.append(summarize(rows, ds))
        print(f"  Done.\n")

    out_dir = Path(__file__).resolve().parent / "baseline_analysis"
    out_dir.mkdir(exist_ok=True)

    write_per_sample_csv(all_rows, out_dir / "per_sample_metrics.csv")
    write_summary_csv(summaries, out_dir / "per_dataset_summary.csv")
    write_summary_md(summaries, out_dir / "per_dataset_summary.md")

    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
