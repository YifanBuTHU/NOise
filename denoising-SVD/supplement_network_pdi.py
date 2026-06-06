from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from svd_denoising.data import SPLIT_GROUPS
from svd_denoising.model import UnetN2N
from svd_denoising.pdi import export_group_network_pdi
from svd_denoising.paths import DEFAULT_FULL_DATA_ROOT, project_relative_path


def supplement_epochs(run_dir: Path, epochs: list[int], device: torch.device, dynamic_range_db: float = 30.0) -> None:
    for epoch in epochs:
        ckpt_path = run_dir / "checkpoints" / f"model_epoch{epoch}.pth"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}, skipping epoch {epoch}")
            continue

        pdi_dir = run_dir / "network_pdi" / f"epoch_{epoch}"
        if (pdi_dir / "manifest.json").exists():
            print(f"Epoch {epoch} PDI already exists at {pdi_dir}, skipping")
            continue

        print(f"Loading checkpoint: {ckpt_path}")
        model = UnetN2N(in_channels=2, out_channels=2).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

        pdi_dir.mkdir(parents=True, exist_ok=True)
        pdi_records = []
        group_to_split = {group: split for split in ("train", "test") for group in SPLIT_GROUPS[split]}
        print(f"Exporting network PDI maps at epoch {epoch} with {dynamic_range_db}dB...")
        for group in SPLIT_GROUPS["train"] + SPLIT_GROUPS["test"]:
            print(f"  {group}: denoise full SVD frames, accumulate power")
            record = export_group_network_pdi(
                model,
                DEFAULT_FULL_DATA_ROOT,
                group,
                pdi_dir,
                device,
                scale_percentile=99.9,
                batch_size=4,
                dynamic_range_db=dynamic_range_db,
            )
            record["split"] = group_to_split[group]
            pdi_records.append(record)
        with (pdi_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "formula": "PDI = sum_t(abs(network(full_svd_t)) ** 2)",
                    "display_dynamic_range_db": dynamic_range_db,
                    "full_data_root": DEFAULT_FULL_DATA_ROOT,
                    "groups": pdi_records,
                },
                handle,
                indent=2,
            )
        print(f"Network PDI saved to {project_relative_path(pdi_dir)}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python supplement_network_pdi.py <run_dir> [epochs...]")
        print("Example: python supplement_network_pdi.py experiments/complex_n2n/20260528-215406 20 40")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        epochs = [int(e) for e in sys.argv[2:]]
    else:
        epochs = [20, 40]

    if not torch.cuda.is_available():
        print("Error: CUDA unavailable", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")
    supplement_epochs(run_dir, epochs, device)


if __name__ == "__main__":
    main()
