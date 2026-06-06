from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from svd_denoising.data import SPLIT_GROUPS, channels_to_complex, make_datasets
from svd_denoising.model import UnetN2N
from svd_denoising.pdi import export_group_network_pdi, export_group_noisy_pdi
from svd_denoising.paths import DEFAULT_DATA_ROOT, DEFAULT_FULL_DATA_ROOT, project_path, project_relative_path


DEFAULT_EXP_DIR = "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train complex SVD Noise2Noise model")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--exp-dir", type=str, default=DEFAULT_EXP_DIR)
    parser.add_argument("--exp-name", type=str, default="complex_n2n")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--scale-percentile", type=float, default=99.9)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt-freq", type=int, default=10)
    parser.add_argument("--preview-freq", type=int, default=10)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--comparison-freq", type=int, default=20)
    parser.add_argument("--comparison-count", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(exp_dir: str, exp_name: str) -> Path:
    run_dir = project_path(Path(exp_dir) / exp_name / time.strftime("%Y%m%d-%H%M%S"))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "previews").mkdir(parents=True, exist_ok=True)
    return run_dir


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _original_magnitude(sample: dict) -> np.ndarray:
    scale = float(sample["scale"])
    original = channels_to_complex(sample["input"].numpy()) * scale
    return np.abs(original)


def select_diverse_test_samples(dataset, count: int) -> list[dict]:
    scored = []
    for index in range(len(dataset)):
        sample = dataset[index]
        mag = _original_magnitude(sample)
        scored.append((float(mag.mean()), index))

    scored.sort(key=lambda item: item[0])
    if count <= 1:
        selected = [scored[len(scored) // 2][1]]
    else:
        positions = np.linspace(0, len(scored) - 1, count).round().astype(int)
        selected = [scored[pos][1] for pos in positions]

    samples = []
    for index in selected:
        sample = dataset[index]
        sample["dataset_index"] = index
        samples.append(sample)
    return samples


def _stack_fixed_samples(samples: list[dict], device: torch.device) -> dict:
    return {
        "input": torch.stack([sample["input"] for sample in samples]).to(device),
        "scale": np.array([float(sample["scale"]) for sample in samples], dtype=np.float32),
        "labels": [
            f"{sample['group']}_{int(sample['frame_index']):04d}_y{int(sample['y'])}_x{int(sample['x'])}"
            for sample in samples
        ],
    }


@torch.no_grad()
def capture_comparison_row(model, fixed_batch: dict, epoch: int) -> dict:
    model.eval()
    outputs = model(fixed_batch["input"]).cpu().numpy()
    inputs = fixed_batch["input"].cpu().numpy()
    scales = fixed_batch["scale"]

    originals = []
    predictions = []
    for idx in range(outputs.shape[0]):
        scale = float(scales[idx])
        originals.append(np.abs(channels_to_complex(inputs[idx]) * scale))
        predictions.append(np.abs(channels_to_complex(outputs[idx]) * scale))
    return {
        "epoch": epoch,
        "originals": originals,
        "predictions": predictions,
    }


def save_comparison_grid(rows: list[dict], labels: list[str], save_path: Path) -> None:
    if not rows:
        return

    n_rows = len(rows)
    n_samples = len(labels)
    fig, axes = plt.subplots(n_rows, n_samples * 2, figsize=(3.0 * n_samples * 2, 3.0 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, n_samples * 2)

    for row_idx, row in enumerate(rows):
        for sample_idx, label in enumerate(labels):
            original = row["originals"][sample_idx]
            pred = row["predictions"][sample_idx]
            vmax = max(float(np.percentile(original, 99.5)), float(np.percentile(pred, 99.5)), 1e-8)

            ax = axes[row_idx, sample_idx * 2]
            ax.imshow(original, cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(f"epoch {row['epoch']} original\n{label}")
            ax.axis("off")

            ax = axes[row_idx, sample_idx * 2 + 1]
            ax.imshow(pred, cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(f"epoch {row['epoch']} predict\n{label}")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_one_epoch(model, loader, optimizer, device, max_batches: int = 0) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    value_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets, reduction="mean")
        loss.backward()
        optimizer.step()

        loss_sum += F.mse_loss(outputs.detach(), targets, reduction="sum").item()
        value_count += targets.numel()
        if max_batches and batch_idx >= max_batches:
            break
    mse = loss_sum / value_count
    return mse, float(np.sqrt(mse))


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 0) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    value_count = 0
    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        outputs = model(inputs)
        loss_sum += F.mse_loss(outputs, targets, reduction="sum").item()
        value_count += targets.numel()
        if max_batches and batch_idx >= max_batches:
            break
    mse = loss_sum / value_count
    return mse, float(np.sqrt(mse))


@torch.no_grad()
def save_preview(model, loader, device, preview_dir: Path, epoch: int, count: int) -> None:
    model.eval()
    batch = next(iter(loader))
    inputs = batch["input"].to(device)
    outputs = model(inputs).cpu().numpy()
    inputs = batch["input"].numpy()
    targets = batch["target"].numpy()
    scales = batch["scale"].numpy()

    for idx in range(min(count, outputs.shape[0])):
        scale = float(scales[idx])
        pred = channels_to_complex(outputs[idx]) * scale
        inp = channels_to_complex(inputs[idx]) * scale
        target = channels_to_complex(targets[idx]) * scale
        stem = f"epoch{epoch:04d}_sample{idx}"
        np.save(preview_dir / f"{stem}_pred.npy", pred.astype(np.complex64))

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, arr, title in zip(axes, (inp, pred, target), ("input", "pred", "target")):
            ax.imshow(np.abs(arr), cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(preview_dir / f"{stem}_magnitude.png", dpi=150)
        plt.close(fig)


def write_args(run_dir: Path, args: argparse.Namespace, dataset_lengths: dict[str, int], device: torch.device, model: UnetN2N) -> None:
    n_params, n_conv_layers = model.model_size
    payload = vars(args).copy()
    payload.update(
        {
            "run_dir": str(project_relative_path(run_dir)),
            "dataset_lengths": dataset_lengths,
            "split_groups": SPLIT_GROUPS,
            "device": str(device),
            "n_params": n_params,
            "n_conv_layers": n_conv_layers,
        }
    )
    with (run_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    import os
    import sys

    if os.environ.get("CONDA_DEFAULT_ENV") != "3DUS":
        print(f"Error: 必须使用 3DUS anaconda 虚拟环境。当前环境: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}", file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        print("Error: CUDA 不可用", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.cuda}")

    datasets = make_datasets(
        args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        sample_interval=args.sample_interval,
        scale_percentile=args.scale_percentile,
    )
    dataset_lengths = {name: len(dataset) for name, dataset in datasets.items()}
    print(f"Dataset lengths: {dataset_lengths}")

    train_loader = make_loader(datasets["train"], args.batch_size, True, args.num_workers)
    test_loader = make_loader(datasets["test"], args.test_batch_size, False, args.num_workers)
    gen_loader = make_loader(datasets["generalization"], args.test_batch_size, False, args.num_workers)

    model = UnetN2N(in_channels=2, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))

    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        run_dir = resume_path.parent.parent
        ckpt = torch.load(resume_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            model.load_state_dict(ckpt)
            start_epoch = int(resume_path.stem.replace("model_epoch", "")) + 1
        print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")
        (run_dir / "previews").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(args.exp_dir, args.exp_name)
        write_args(run_dir, args, dataset_lengths, device, model)

    fixed_samples = select_diverse_test_samples(datasets["test"], args.comparison_count)
    fixed_batch = _stack_fixed_samples(fixed_samples, device)
    comparison_rows = []
    print(f"Fixed comparison samples: {fixed_batch['labels']}")

    def _export_network_pdi_at_epoch(epoch: int | None = None, dynamic_range_db: float = 30.0) -> None:
        pdi_dir = run_dir / "network_pdi" if epoch is None else run_dir / "network_pdi" / f"epoch_{epoch}"
        pdi_dir.mkdir(parents=True, exist_ok=True)
        group_to_split = {group: split for split in ("train", "test") for group in SPLIT_GROUPS[split]}

        print(f"Exporting network PDI maps{' at epoch ' + str(epoch) if epoch else ''}...")
        pdi_records = []
        for group in SPLIT_GROUPS["train"] + SPLIT_GROUPS["test"]:
            print(f"  {group}: denoise full SVD frames, accumulate power")
            record = export_group_network_pdi(
                model,
                DEFAULT_FULL_DATA_ROOT,
                group,
                pdi_dir,
                device,
                scale_percentile=args.scale_percentile,
                batch_size=max(1, args.test_batch_size),
                dynamic_range_db=dynamic_range_db,
            )
            record["split"] = group_to_split[group]
            pdi_records.append(record)

        print(f"Exporting noisy SVD PDI maps{' at epoch ' + str(epoch) if epoch else ''}...")
        noisy_records = []
        for group in SPLIT_GROUPS["train"] + SPLIT_GROUPS["test"]:
            print(f"  {group}: compute noisy SVD PDI with noise equalization")
            noisy_record = export_group_noisy_pdi(
                DEFAULT_FULL_DATA_ROOT,
                group,
                pdi_dir,
                dynamic_range_db=dynamic_range_db,
            )
            noisy_record["split"] = group_to_split[group]
            noisy_records.append(noisy_record)

        with (pdi_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "formula": "PDI = sum_t(abs(network(full_svd_t)) ** 2)",
                    "display_dynamic_range_db": dynamic_range_db,
                    "full_data_root": DEFAULT_FULL_DATA_ROOT,
                    "groups": pdi_records + noisy_records,
                },
                handle,
                indent=2,
            )
        print(f"Network and noisy PDI saved to {project_relative_path(pdi_dir)}")

    metrics_path = run_dir / "metrics.csv"
    mode = "a" if args.resume else "w"
    with metrics_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_mse",
                "train_rmse",
                "test_mse",
                "test_rmse",
                "generalization_mse",
                "generalization_rmse",
            ],
        )
        if not args.resume:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            train_mse, train_rmse = train_one_epoch(
                model, train_loader, optimizer, device, max_batches=args.max_train_batches
            )
            test_mse, test_rmse = evaluate(model, test_loader, device, max_batches=args.max_eval_batches)
            gen_mse, gen_rmse = evaluate(model, gen_loader, device, max_batches=args.max_eval_batches)

            row = {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_rmse": train_rmse,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "generalization_mse": gen_mse,
                "generalization_rmse": gen_rmse,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"train_rmse={train_rmse:.6f} test_rmse={test_rmse:.6f} gen_rmse={gen_rmse:.6f}"
            )

            if epoch % args.ckpt_freq == 0 or epoch == args.epochs:
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                    run_dir / "checkpoints" / f"model_epoch{epoch}.pth",
                )
            if args.preview_freq and (epoch % args.preview_freq == 0 or epoch == args.epochs):
                save_preview(model, test_loader, device, run_dir / "previews", epoch, args.preview_count)
            if args.comparison_freq and (epoch % args.comparison_freq == 0 or epoch == args.epochs):
                comparison_rows.append(capture_comparison_row(model, fixed_batch, epoch))
            if epoch % 20 == 0 or epoch == args.epochs:
                _export_network_pdi_at_epoch(epoch, dynamic_range_db=30.0)

    save_comparison_grid(
        comparison_rows,
        fixed_batch["labels"],
        run_dir / "previews" / f"test_comparison_every{args.comparison_freq}_epochs.png",
    )

    # Final network PDI and noisy PDI export
    _export_network_pdi_at_epoch()

    print(f"Run saved to {project_relative_path(run_dir)}")


if __name__ == "__main__":
    main()
