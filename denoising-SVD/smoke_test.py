from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from svd_denoising.data import make_datasets
from svd_denoising.model import UnetN2N
from svd_denoising.paths import DEFAULT_DATA_ROOT, project_path, project_relative_path


def main() -> None:
    import os
    import sys

    if os.environ.get("CONDA_DEFAULT_ENV") != "3DUS":
        print(f"Error: 必须使用 3DUS anaconda 虚拟环境。当前环境: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("Error: CUDA 不可用", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")

    datasets = make_datasets(DEFAULT_DATA_ROOT)
    lengths = {name: len(dataset) for name, dataset in datasets.items()}
    expected = {"train": 12600, "test": 900, "generalization": 1350}
    if lengths != expected:
        raise AssertionError(f"Unexpected dataset lengths: {lengths}, expected {expected}")

    loader = DataLoader(datasets["train"], batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)
    if tuple(inputs.shape) != (1, 2, 256, 256):
        raise AssertionError(f"Unexpected input shape: {tuple(inputs.shape)}")
    if not torch.isfinite(batch["scale"]).all():
        raise AssertionError("Scale contains non-finite values")

    model = UnetN2N().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    if outputs.shape != targets.shape:
        raise AssertionError(f"Unexpected output shape: {tuple(outputs.shape)}")
    loss = F.mse_loss(outputs, targets)
    if not torch.isfinite(loss):
        raise AssertionError("Loss is not finite")
    loss.backward()
    optimizer.step()

    run_dir = project_path(Path("experiments") / f"smoke_{time.strftime('%Y%m%d-%H%M%S')}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), ckpt_dir / "model_epoch1.pth")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "train_mse": float(loss.detach().cpu())})

    print(f"smoke ok: lengths={lengths}, loss={float(loss.detach().cpu()):.6f}, run_dir={project_relative_path(run_dir)}")


if __name__ == "__main__":
    main()
