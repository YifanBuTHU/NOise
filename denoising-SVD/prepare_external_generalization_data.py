from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from svd_denoising.paths import DATA_ROOT, DEFAULT_GENERALIZATION_FULL_ROOT


DEFAULT_SOURCE_ROOT = DATA_ROOT
TILE_SIZE = 512

CARE_GROUPS = {
    "train_low_snr": "CARE2D_Train_LowSNR",
    "train_high_snr": "CARE2D_Train_HighSNR",
    "test_low_snr": "CARE2D_Test_LowSNR",
    "test_high_snr": "CARE2D_Test_HighSNR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MRC and CARE 2D images as complex full-frame N2N generalization data")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_GENERALIZATION_FULL_ROOT)
    parser.add_argument("--tile-size", default=TILE_SIZE, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_tiles(image: np.ndarray, tile_size: int) -> Iterator[tuple[int, int, np.ndarray]]:
    height, width = image.shape
    for y in range(0, height - tile_size + 1, tile_size):
        for x in range(0, width - tile_size + 1, tile_size):
            yield y, x, image[y : y + tile_size, x : x + tile_size]


def save_complex_frame(path: Path, image: np.ndarray) -> None:
    frame = np.ascontiguousarray(image.astype(np.float32, copy=False)).astype(np.complex64)
    np.save(path, frame)


def ensure_output_root(path: Path, dry_run: bool) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output root already contains files: {path}")
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def read_mrc_image(path: Path) -> tuple[np.memmap, dict[str, int | str]]:
    with path.open("rb") as handle:
        header = handle.read(1024)

    nx, ny, nz, mode = struct.unpack("<4i", header[:16])
    nsymbt = struct.unpack("<i", header[92:96])[0]
    if mode != 2 or nz != 1:
        raise ValueError(f"{path} uses MRC mode={mode}, nz={nz}; expected one float32 section")

    offset = 1024 + nsymbt
    data_count = (path.stat().st_size - offset) // np.dtype(np.float32).itemsize
    expected_count = nx * ny
    inferred = False
    if data_count != expected_count:
        factor = math.sqrt(data_count / expected_count)
        rounded = round(factor)
        if abs(factor - rounded) > 1e-9:
            raise ValueError(f"{path} data count does not match header shape")
        nx *= rounded
        ny *= rounded
        inferred = True

    image = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(ny, nx))
    meta = {
        "source": str(path),
        "height": ny,
        "width": nx,
        "header_shape_inferred_from_file_size": str(inferred),
    }
    return image, meta


def group_for_mrc(path: Path) -> str:
    if path.name.startswith("stack_"):
        return "MRC_stack_DW"
    if path.name.startswith("Yt_"):
        return "MRC_Yt_105kx"
    return "MRC_other"


def prepare_mrc(source_root: Path, output_root: Path, tile_size: int, dry_run: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    group_counts: dict[str, int] = {}

    for source_path in sorted((source_root / "MRC").glob("*.mrc")):
        image, meta = read_mrc_image(source_path)
        group = group_for_mrc(source_path)
        group_dir = output_root / group
        if not dry_run:
            group_dir.mkdir(parents=True, exist_ok=True)

        for y, x, tile in iter_tiles(image, tile_size):
            index = group_counts.get(group, 0)
            output_name = f"{group}_full_{index:04d}.npy"
            if not dry_run:
                save_complex_frame(group_dir / output_name, tile)
            records.append(
                {
                    "group": group,
                    "output": output_name,
                    "source": meta["source"],
                    "source_height": meta["height"],
                    "source_width": meta["width"],
                    "header_shape_inferred_from_file_size": meta["header_shape_inferred_from_file_size"],
                    "y": y,
                    "x": x,
                }
            )
            group_counts[group] = index + 1

    return records


def prepare_care(source_root: Path, output_root: Path, tile_size: int, dry_run: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    group_counts: dict[str, int] = {}

    for folder_name, group in CARE_GROUPS.items():
        folder = source_root / "CARE_2D" / folder_name
        group_dir = output_root / group
        if not dry_run:
            group_dir.mkdir(parents=True, exist_ok=True)

        for source_path in sorted(folder.glob("*.tif")):
            with Image.open(source_path) as image_file:
                image = np.asarray(image_file, dtype=np.float32)
            if image.ndim != 2:
                raise ValueError(f"{source_path} has shape {image.shape}; expected a 2D image")

            for y, x, tile in iter_tiles(image, tile_size):
                index = group_counts.get(group, 0)
                output_name = f"{group}_full_{index:04d}.npy"
                if not dry_run:
                    save_complex_frame(group_dir / output_name, tile)
                records.append(
                    {
                        "group": group,
                        "output": output_name,
                        "source": str(source_path),
                        "source_height": int(image.shape[0]),
                        "source_width": int(image.shape[1]),
                        "y": y,
                        "x": x,
                    }
                )
                group_counts[group] = index + 1

    return records


def summarize(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        group = str(record["group"])
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    ensure_output_root(output_root, args.dry_run)

    records = []
    records.extend(prepare_mrc(source_root, output_root, args.tile_size, args.dry_run))
    records.extend(prepare_care(source_root, output_root, args.tile_size, args.dry_run))

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "tile_size": args.tile_size,
        "dtype": "complex64",
        "imaginary_channel": "zeros",
        "groups": summarize(records),
        "records": records,
    }
    if not args.dry_run:
        with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    print(json.dumps({k: manifest[k] for k in ("output_root", "tile_size", "dtype", "groups")}, indent=2))


if __name__ == "__main__":
    main()
