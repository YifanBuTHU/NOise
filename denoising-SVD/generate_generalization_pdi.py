from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from svd_denoising.model import UnetN2N
from svd_denoising.pdi import export_group_network_pdi, export_group_noisy_pdi
from svd_denoising.paths import DEFAULT_FULL_DATA_ROOT, project_relative_path


def main() -> None:
    run_dir = Path("experiments/complex_n2n/20260528-215406")
    ckpt_path = run_dir / "checkpoints" / "model_epoch100.pth"
    output_dir = run_dir / "network_pdi" / "generalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = ["HalfRing_M3", "P1", "P2", "P3"]
    scale_percentile = 99.9
    batch_size = 4
    dynamic_range_db = 30.0

    if not torch.cuda.is_available():
        print("Error: CUDA unavailable", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {ckpt_path}")
    model = UnetN2N(in_channels=2, out_channels=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    pdi_records = []

    for group in groups:
        print(f"\n[{group}] Exporting noisy SVD PDI...")
        noisy_record = export_group_noisy_pdi(
            DEFAULT_FULL_DATA_ROOT,
            group,
            output_dir,
            dynamic_range_db=dynamic_range_db,
        )
        pdi_records.append(noisy_record)
        print(f"  Saved: {noisy_record['pdi']}")

        print(f"[{group}] Exporting network PDI...")
        network_record = export_group_network_pdi(
            model,
            DEFAULT_FULL_DATA_ROOT,
            group,
            output_dir,
            device,
            scale_percentile=scale_percentile,
            batch_size=batch_size,
            dynamic_range_db=dynamic_range_db,
        )
        network_record["split"] = "generalization"
        pdi_records.append(network_record)
        print(f"  Saved: {network_record['pdi']}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "formula": "PDI = sum_t(abs(network(full_svd_t)) ** 2)",
                "display_dynamic_range_db": dynamic_range_db,
                "full_data_root": str(DEFAULT_FULL_DATA_ROOT),
                "groups": pdi_records,
            },
            handle,
            indent=2,
        )
    print(f"\nManifest saved to {project_relative_path(manifest_path)}")


if __name__ == "__main__":
    main()
