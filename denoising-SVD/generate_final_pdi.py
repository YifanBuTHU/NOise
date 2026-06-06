from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from svd_denoising.data import SPLIT_GROUPS
from svd_denoising.model import UnetN2N
from svd_denoising.pdi import export_group_network_pdi, export_group_noisy_pdi
from svd_denoising.paths import DEFAULT_FULL_DATA_ROOT, project_relative_path


def generate_final_pdi(run_dir: Path, device: torch.device, dynamic_range_db: float = 30.0) -> None:
    ckpt_path = run_dir / "checkpoints" / "model_epoch100.pth"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    pdi_dir = run_dir / "network_pdi"
    if (pdi_dir / "manifest.json").exists():
        print(f"Final PDI already exists at {pdi_dir}, skipping")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    model = UnetN2N(in_channels=2, out_channels=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    pdi_dir.mkdir(parents=True, exist_ok=True)
    group_to_split = {group: split for split in ("train", "test") for group in SPLIT_GROUPS[split]}

    print(f"Exporting network PDI maps with {dynamic_range_db}dB...")
    pdi_records = []
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

    print(f"Exporting noisy SVD PDI maps with {dynamic_range_db}dB...")
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


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python generate_final_pdi.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    if not torch.cuda.is_available():
        print("Error: CUDA unavailable", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")
    generate_final_pdi(run_dir, device)


if __name__ == "__main__":
    main()
