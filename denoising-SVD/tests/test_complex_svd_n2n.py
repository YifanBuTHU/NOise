import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from svd_denoising.data import ComplexSvdN2NDataset, SPLIT_GROUPS, make_datasets
from svd_denoising.model import UnetN2N
from svd_denoising.pdi import compute_power_doppler, export_group_network_pdi


GROUPS = ("M1", "M2", "M3", "P1", "P2", "P3", "R1", "R2", "R3", "R4", "R5", "R6")


def _write_pair(root, group, index, shape=(8, 8)):
    even_dir = root / group / "even"
    odd_dir = root / group / "odd"
    even_dir.mkdir(parents=True, exist_ok=True)
    odd_dir.mkdir(parents=True, exist_ok=True)

    base = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape) + index + 1
    even = (base + 1j * (base * 0.5)).astype(np.complex64)
    odd = (base * 2 - 1j * base).astype(np.complex64)
    np.save(even_dir / f"{group}_even_{index:04d}.npy", even)
    np.save(odd_dir / f"{group}_odd_{index:04d}.npy", odd)


def _write_full(root, group, index, shape=(8, 8)):
    group_dir = root / group
    group_dir.mkdir(parents=True, exist_ok=True)

    base = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape) + index + 1
    full = (base * 3 + 1j * (base * 0.25)).astype(np.complex64)
    np.save(group_dir / f"{group}_full_{index:04d}.npy", full)


def _write_dataset(root):
    for group in GROUPS:
        for index in range(20):
            _write_pair(root, group, index)


def test_split_lengths_with_interval_patches_and_rotations(tmp_path):
    _write_dataset(tmp_path)

    datasets = make_datasets(tmp_path, patch_size=4, stride=2, sample_interval=10, image_size=8)

    assert SPLIT_GROUPS["train"] == ("M1", "M2", "R1", "R2", "R3", "R4", "R5")
    assert SPLIT_GROUPS["test"] == ("M3", "R6")
    assert SPLIT_GROUPS["generalization"] == ("P1", "P2", "P3")
    assert len(datasets["train"]) == 7 * 2 * 9 * 4
    assert len(datasets["test"]) == 2 * 2 * 9
    assert len(datasets["generalization"]) == 3 * 2 * 9


def test_complex_patch_uses_two_channels_and_shared_input_scale(tmp_path):
    _write_dataset(tmp_path)
    dataset = ComplexSvdN2NDataset(
        tmp_path,
        groups=("M1",),
        patch_size=4,
        stride=2,
        sample_interval=10,
        augment_rotations=False,
        image_size=8,
    )

    sample = dataset[0]
    even = np.load(tmp_path / "M1" / "even" / "M1_even_0000.npy")[:4, :4]
    odd = np.load(tmp_path / "M1" / "odd" / "M1_odd_0000.npy")[:4, :4]
    scale = max(float(np.percentile(np.abs(even), 99.9)), 1e-8)

    assert sample["input"].shape == (2, 4, 4)
    assert sample["target"].shape == (2, 4, 4)
    assert torch.isclose(sample["scale"], torch.tensor(scale, dtype=torch.float32))
    np.testing.assert_allclose(sample["input"][0].numpy(), even.real / scale)
    np.testing.assert_allclose(sample["input"][1].numpy(), even.imag / scale)
    np.testing.assert_allclose(sample["target"][0].numpy(), odd.real / scale)
    np.testing.assert_allclose(sample["target"][1].numpy(), odd.imag / scale)


def test_unet_forward_preserves_complex_channel_shape():
    model = UnetN2N(in_channels=2, out_channels=2)
    x = torch.randn(1, 2, 256, 256)

    y = model(x)

    assert y.shape == x.shape


def test_power_doppler_sums_frame_power_not_coherent_amplitude():
    frames = [
        np.array([[1 + 0j, 1 + 1j]], dtype=np.complex64),
        np.array([[-1 + 0j, 2 + 0j]], dtype=np.complex64),
    ]

    pdi = compute_power_doppler(frames)

    np.testing.assert_allclose(pdi, np.array([[2.0, 6.0]], dtype=np.float32))


def test_export_group_network_pdi_uses_full_frames_directly(tmp_path):
    class IdentityModel(torch.nn.Module):
        def forward(self, x):
            return x

    _write_full(tmp_path, "M1", 0, shape=(8, 8))
    _write_full(tmp_path, "M1", 1, shape=(8, 8))
    output_dir = tmp_path / "pdi"

    record = export_group_network_pdi(
        IdentityModel(),
        tmp_path,
        "M1",
        output_dir,
        torch.device("cpu"),
        scale_percentile=99.9,
        batch_size=2,
        dynamic_range_db=45.0,
    )

    assert record["frame_count"] == 2
    assert (output_dir / "M1_network_pdi.npy").is_file()
    assert (output_dir / "M1_network_pdi_45db.png").is_file()

    expected_frames = []
    for index in range(2):
        expected_frames.append(np.load(tmp_path / "M1" / f"M1_full_{index:04d}.npy"))

    np.testing.assert_allclose(
        np.load(output_dir / "M1_network_pdi.npy"),
        compute_power_doppler(expected_frames),
        rtol=1e-6,
    )
