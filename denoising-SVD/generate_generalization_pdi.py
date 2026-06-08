from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from svd_denoising.model import UnetN2N
from svd_denoising.pdi import export_group_network_pdi, export_group_noisy_pdi
from svd_denoising.paths import DEFAULT_GENERALIZATION_FULL_ROOT, project_path, project_relative_path


DEFAULT_RUN_DIR = "experiments/complex_n2n/20260528-215406"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export N2N PDI maps for external full-frame generalization data")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Training run directory with checkpoints")
    parser.add_argument("--epoch", default=100, type=int, help="Checkpoint epoch to load")
    parser.add_argument("--full-root", default=DEFAULT_GENERALIZATION_FULL_ROOT, help="Root containing group/*.npy frames")
    parser.add_argument("--groups", nargs="*", default=None, help="Group folders to export; defaults to all folders in full-root")
    parser.add_argument("--output-dir", default=None, help="Output directory; defaults to run_dir/network_pdi/external_generalization")
    parser.add_argument("--batch-size", default=1, type=int, help="Inference batch size")
    parser.add_argument("--scale-percentile", default=99.9, type=float)
    parser.add_argument("--dynamic-range-db", default=30.0, type=float)
    parser.add_argument("--cuda", default=0, type=int)
    return parser.parse_args()


def discover_groups(full_root: Path) -> list[str]:
    groups = sorted(path.name for path in full_root.iterdir() if path.is_dir())
    if not groups:
        raise FileNotFoundError(f"No group folders found under {full_root}")
    return groups


def main() -> None:
    args = parse_args()
    run_dir = project_path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / f"model_epoch{args.epoch}.pth"
    full_root = project_path(args.full_root)
    groups = args.groups if args.groups else discover_groups(full_root)
    output_dir = project_path(args.output_dir) if args.output_dir else run_dir / "network_pdi" / "external_generalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Error: CUDA unavailable", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args.cuda}")
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
            full_root,
            group,
            output_dir,
            dynamic_range_db=args.dynamic_range_db,
        )
        pdi_records.append(noisy_record)
        print(f"  Saved: {noisy_record['pdi']}")

        print(f"[{group}] Exporting network PDI...")
        network_record = export_group_network_pdi(
            model,
            full_root,
            group,
            output_dir,
            device,
            scale_percentile=args.scale_percentile,
            batch_size=args.batch_size,
            dynamic_range_db=args.dynamic_range_db,
        )
        network_record["split"] = "external_generalization"
        pdi_records.append(network_record)
        print(f"  Saved: {network_record['pdi']}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "formula": "PDI = sum_t(abs(network(full_svd_t)) ** 2)",
                "display_dynamic_range_db": args.dynamic_range_db,
                "full_data_root": str(full_root),
                "groups": pdi_records,
            },
            handle,
            indent=2,
        )
    print(f"\nManifest saved to {project_relative_path(manifest_path)}")


if __name__ == "__main__":
    main()
