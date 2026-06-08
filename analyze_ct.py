import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

DATA_ROOT = "F:/NOise_data/CT"
OUT_DIR = "F:/NOise_data/N2N_result/CT/analysis/ct_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

gt_files = sorted(glob(os.path.join(DATA_ROOT, "ground_truth_test", "*.hdf5")))
obs_files = sorted(glob(os.path.join(DATA_ROOT, "observation_test", "*.hdf5")))

print(f"GT files: {len(gt_files)}, OBS files: {len(obs_files)}")


def collect_stats(files, name):
    stats = {
        "shapes": [],
        "dtypes": [],
        "mins": [],
        "maxs": [],
        "means": [],
        "stds": [],
        "p01": [],
        "p99": [],
        "samples": [],
    }
    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            data = f['data'][:]
        stats["shapes"].append(data.shape)
        stats["dtypes"].append(data.dtype)
        stats["mins"].append(float(data.min()))
        stats["maxs"].append(float(data.max()))
        stats["means"].append(float(data.mean()))
        stats["stds"].append(float(data.std()))
        stats["p01"].append(float(np.percentile(data, 1)))
        stats["p99"].append(float(np.percentile(data, 99)))
        stats["samples"].append(data.size)
        print(f"  {name} {os.path.basename(fpath)}: shape={data.shape}, dtype={data.dtype}, "
              f"min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}")
    return stats


gt_stats = collect_stats(gt_files, "GT")
obs_stats = collect_stats(obs_files, "OBS")


def print_summary(stats, label):
    print(f"\n=== {label} Summary ===")
    shapes = stats["shapes"]
    unique_shapes = set(shapes)
    print(f"Shapes: {unique_shapes}")
    print(f"Count: {len(shapes)}")
    print(f"Min range: [{min(stats['mins']):.4f}, {max(stats['mins']):.4f}]")
    print(f"Max range: [{min(stats['maxs']):.4f}, {max(stats['maxs']):.4f}]")
    print(f"Mean range: [{min(stats['means']):.4f}, {max(stats['means']):.4f}]")
    print(f"Std range:  [{min(stats['stds']):.4f}, {max(stats['stds']):.4f}]")
    print(f"1% percentile range: [{min(stats['p01']):.4f}, {max(stats['p01']):.4f}]")
    print(f"99% percentile range: [{min(stats['p99']):.4f}, {max(stats['p99']):.4f}]")
    print(f"Total voxels: {sum(stats['samples'])}")


print_summary(gt_stats, "Ground Truth")
print_summary(obs_stats, "Observation")

# Save summary report
report_path = os.path.join(OUT_DIR, "summary.txt")
with open(report_path, "w") as rep:
    rep.write("CT Dataset Analysis Summary\n")
    rep.write("="*50 + "\n\n")
    rep.write(f"Number of samples: {len(gt_files)}\n")
    rep.write(f"GT shape: {gt_stats['shapes'][0]}, dtype: {gt_stats['dtypes'][0]}\n")
    rep.write(f"OBS shape: {obs_stats['shapes'][0]}, dtype: {obs_stats['dtypes'][0]}\n\n")
    rep.write("Ground Truth Statistics:\n")
    rep.write(f"  min: [{min(gt_stats['mins']):.4f}, {max(gt_stats['mins']):.4f}]\n")
    rep.write(f"  max: [{min(gt_stats['maxs']):.4f}, {max(gt_stats['maxs']):.4f}]\n")
    rep.write(f"  mean: [{min(gt_stats['means']):.4f}, {max(gt_stats['means']):.4f}]\n")
    rep.write(f"  std: [{min(gt_stats['stds']):.4f}, {max(gt_stats['stds']):.4f}]\n")
    rep.write(f"  p01: [{min(gt_stats['p01']):.4f}, {max(gt_stats['p01']):.4f}]\n")
    rep.write(f"  p99: [{min(gt_stats['p99']):.4f}, {max(gt_stats['p99']):.4f}]\n\n")
    rep.write("Observation Statistics:\n")
    rep.write(f"  min: [{min(obs_stats['mins']):.4f}, {max(obs_stats['mins']):.4f}]\n")
    rep.write(f"  max: [{min(obs_stats['maxs']):.4f}, {max(obs_stats['maxs']):.4f}]\n")
    rep.write(f"  mean: [{min(obs_stats['means']):.4f}, {max(obs_stats['means']):.4f}]\n")
    rep.write(f"  std: [{min(obs_stats['stds']):.4f}, {max(obs_stats['stds']):.4f}]\n")
    rep.write(f"  p01: [{min(obs_stats['p01']):.4f}, {max(obs_stats['p01']):.4f}]\n")
    rep.write(f"  p99: [{min(obs_stats['p99']):.4f}, {max(obs_stats['p99']):.4f}]\n")
print(f"Summary saved to {report_path}")

# Plot distribution of per-sample statistics
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].hist(gt_stats['mins'], bins=20, color='blue', alpha=0.7, label='GT min')
axes[0, 0].hist(obs_stats['mins'], bins=20, color='red', alpha=0.7, label='OBS min')
axes[0, 0].set_title("Min Value Distribution")
axes[0, 0].legend()

axes[0, 1].hist(gt_stats['maxs'], bins=20, color='blue', alpha=0.7, label='GT max')
axes[0, 1].hist(obs_stats['maxs'], bins=20, color='red', alpha=0.7, label='OBS max')
axes[0, 1].set_title("Max Value Distribution")
axes[0, 1].legend()

axes[0, 2].hist(gt_stats['means'], bins=20, color='blue', alpha=0.7, label='GT mean')
axes[0, 2].hist(obs_stats['means'], bins=20, color='red', alpha=0.7, label='OBS mean')
axes[0, 2].set_title("Mean Value Distribution")
axes[0, 2].legend()

axes[1, 0].hist(gt_stats['stds'], bins=20, color='blue', alpha=0.7, label='GT std')
axes[1, 0].hist(obs_stats['stds'], bins=20, color='red', alpha=0.7, label='OBS std')
axes[1, 0].set_title("Std Value Distribution")
axes[1, 0].legend()

axes[1, 1].hist(gt_stats['p99'], bins=20, color='blue', alpha=0.7, label='GT p99')
axes[1, 1].hist(obs_stats['p99'], bins=20, color='red', alpha=0.7, label='OBS p99')
axes[1, 1].set_title("99th Percentile Distribution")
axes[1, 1].legend()

axes[1, 2].hist(gt_stats['p01'], bins=20, color='blue', alpha=0.7, label='GT p01')
axes[1, 2].hist(obs_stats['p01'], bins=20, color='red', alpha=0.7, label='OBS p01')
axes[1, 2].set_title("1st Percentile Distribution")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stat_distributions.png"), dpi=150)
plt.close()
print("Saved stat_distributions.png")

# Global histogram with sampled data (4 files each)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# GT histogram
sample_gt = gt_files[::max(1, len(gt_files)//4)]
all_gt_min, all_gt_max = min(gt_stats['mins']), max(gt_stats['maxs'])
bins_gt = np.linspace(all_gt_min, all_gt_max, 201)
hist_gt = np.zeros(200)
for fpath in sample_gt:
    with h5py.File(fpath, 'r') as f:
        data = f['data'][:].ravel()
    h, _ = np.histogram(data, bins=bins_gt)
    hist_gt += h
axes[0].stairs(hist_gt, bins_gt, fill=True, alpha=0.7)
axes[0].set_title(f"GT Value Histogram (sampled {len(sample_gt)} files)")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Count")

# OBS histogram
sample_obs = obs_files[::max(1, len(obs_files)//4)]
all_obs_min, all_obs_max = min(obs_stats['mins']), max(obs_stats['maxs'])
bins_obs = np.linspace(all_obs_min, all_obs_max, 201)
hist_obs = np.zeros(200)
for fpath in sample_obs:
    with h5py.File(fpath, 'r') as f:
        data = f['data'][:].ravel()
    h, _ = np.histogram(data, bins=bins_obs)
    hist_obs += h
axes[1].stairs(hist_obs, bins_obs, fill=True, alpha=0.7, color='orange')
axes[1].set_title(f"OBS Value Histogram (sampled {len(sample_obs)} files)")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "value_histograms.png"), dpi=150)
plt.close()
print("Saved value_histograms.png")

# Show typical slices from one representative sample (first file)
rep_gt_path = gt_files[0]
rep_obs_path = obs_files[0]

with h5py.File(rep_gt_path, 'r') as f:
    gt_vol = f['data'][:]
with h5py.File(rep_obs_path, 'r') as f:
    obs_vol = f['data'][:]

print(f"Representative GT shape: {gt_vol.shape}")
print(f"Representative OBS shape: {obs_vol.shape}")

# GT axial, coronal, sagittal middle slices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

z_mid = gt_vol.shape[0] // 2
y_mid = gt_vol.shape[1] // 2
x_mid = gt_vol.shape[2] // 2

axes[0, 0].imshow(gt_vol[z_mid], cmap='gray')
axes[0, 0].set_title(f"GT Axial (z={z_mid})")
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_vol[:, y_mid, :], cmap='gray')
axes[0, 1].set_title(f"GT Coronal (y={y_mid})")
axes[0, 1].axis('off')

axes[0, 2].imshow(gt_vol[:, :, x_mid], cmap='gray')
axes[0, 2].set_title(f"GT Sagittal (x={x_mid})")
axes[0, 2].axis('off')

# OBS slices
z_mid_o = obs_vol.shape[0] // 2
y_mid_o = obs_vol.shape[1] // 2
x_mid_o = obs_vol.shape[2] // 2

axes[1, 0].imshow(obs_vol[z_mid_o], cmap='gray')
axes[1, 0].set_title(f"OBS slice 0 (z={z_mid_o})")
axes[1, 0].axis('off')

axes[1, 1].imshow(obs_vol[:, y_mid_o, :], cmap='gray')
axes[1, 1].set_title(f"OBS slice 1 (y={y_mid_o})")
axes[1, 1].axis('off')

axes[1, 2].imshow(obs_vol[:, :, x_mid_o], cmap='gray')
axes[1, 2].set_title(f"OBS slice 2 (x={x_mid_o})")
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "typical_slices.png"), dpi=150)
plt.close()
print("Saved typical_slices.png")

# Show multiple slices across z for GT and OBS side by side
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, z in enumerate(np.linspace(20, gt_vol.shape[0]-20, 5, dtype=int)):
    axes[0, i].imshow(gt_vol[z], cmap='gray', vmin=gt_vol.min(), vmax=gt_vol.max())
    axes[0, i].set_title(f"GT z={z}")
    axes[0, i].axis('off')

for i, z in enumerate(np.linspace(20, obs_vol.shape[0]-20, 5, dtype=int)):
    axes[1, i].imshow(obs_vol[z], cmap='gray', vmin=obs_vol.min(), vmax=obs_vol.max())
    axes[1, i].set_title(f"OBS z={z}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "z_progression.png"), dpi=150)
plt.close()
print("Saved z_progression.png")

# Show GT with different windowing based on actual data range
# Data is normalized [0,1], not HU; show with different contrast windows
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
p1, p99 = np.percentile(gt_vol, [1, 99])
p5, p95 = np.percentile(gt_vol, [5, 95])

axes[0].imshow(gt_vol[z_mid], cmap='gray', vmin=gt_vol.min(), vmax=gt_vol.max())
axes[0].set_title(f"GT Full range [{gt_vol.min():.3f}, {gt_vol.max():.3f}]")
axes[0].axis('off')

axes[1].imshow(gt_vol[z_mid], cmap='gray', vmin=p1, vmax=p99)
axes[1].set_title(f"GT 1%-99% window [{p1:.3f}, {p99:.3f}]")
axes[1].axis('off')

axes[2].imshow(gt_vol[z_mid], cmap='gray', vmin=p5, vmax=p95)
axes[2].set_title(f"GT 5%-95% window [{p5:.3f}, {p95:.3f}]")
axes[2].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gt_windowing.png"), dpi=150)
plt.close()
print("Saved gt_windowing.png")

# OBS sinogram-like visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# show a few z slices
for idx, z in enumerate([0, obs_vol.shape[0]//4, obs_vol.shape[0]//2, obs_vol.shape[0]*3//4, obs_vol.shape[0]-1]):
    if idx >= 5:
        break
    ax = axes[idx // 3, idx % 3]
    im = ax.imshow(obs_vol[z], cmap='gray', aspect='auto')
    ax.set_title(f"OBS projection z={z}")
    ax.set_xlabel("detector")
    ax.set_ylabel("angle/view")
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[-1, -1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "obs_projections.png"), dpi=150)
plt.close()
print("Saved obs_projections.png")

# Per-slice statistics for one GT volume (to show consistency)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
gt_slice_means = [gt_vol[i].mean() for i in range(gt_vol.shape[0])]
gt_slice_stds = [gt_vol[i].std() for i in range(gt_vol.shape[0])]
gt_slice_ranges = [gt_vol[i].max() - gt_vol[i].min() for i in range(gt_vol.shape[0])]

axes[0].plot(gt_slice_means)
axes[0].set_title("GT Per-slice Mean")
axes[0].set_xlabel("Slice index")
axes[0].set_ylabel("Mean")

axes[1].plot(gt_slice_stds)
axes[1].set_title("GT Per-slice Std")
axes[1].set_xlabel("Slice index")
axes[1].set_ylabel("Std")

axes[2].plot(gt_slice_ranges)
axes[2].set_title("GT Per-slice Range (max-min)")
axes[2].set_xlabel("Slice index")
axes[2].set_ylabel("Range")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gt_slice_stats.png"), dpi=150)
plt.close()
print("Saved gt_slice_stats.png")

print("\nAll outputs saved to:", OUT_DIR)
