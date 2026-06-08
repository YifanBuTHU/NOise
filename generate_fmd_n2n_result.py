"""Generate FMD N2N results into N2N_result format."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add fluorescence project to path
FLUORE_DIR = Path(__file__).resolve().parent / "denoising-fluorescence" / "denoising"
sys.path.insert(0, str(FLUORE_DIR))

from models.unet import UnetN2Nv2
from utils.data_loader import fluore_to_tensor


def fluore_to_tensor_no_crop(pic):
    """Convert PIL Image to tensor without any spatial cropping."""
    if not hasattr(pic, 'mode'):
        raise TypeError(f'pic should be PIL Image. Got {type(pic)}')

    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)

    if nchannel == 1:
        img = img.squeeze(-1).unsqueeze(0)
    elif pic.mode in ('RGB', 'RGBA'):
        ori_dtype = img.dtype
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
        img = img.to(ori_dtype)
    else:
        raise TypeError(f'Unsupported image type {pic.mode}')

    return img


def preprocess(img):
    """Same normalization as training: [0,255] -> [-0.5, 0.5]."""
    t = fluore_to_tensor_no_crop(img)
    return t.float().div(255.0).sub(0.5)


def postprocess(tensor):
    """Inverse normalization: [-0.5, 0.5] -> [0, 255], clip, uint8."""
    arr = tensor.add(0.5).mul(255.0).clamp(0, 255).cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    return arr.astype(np.uint8)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    fmd_root = Path(r"F:\NOise_data\FMD")
    test_mix_raw = fmd_root / "test_mix" / "raw"
    test_mix_gt = fmd_root / "test_mix" / "gt"
    result_dir = Path(r"F:\NOise_data\N2N_result\FMD\n2n_raw_unetv2_256_e300")
    samples_dir = result_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Model checkpoint
    ckpt_path = (
        Path(__file__).resolve().parent
        / "denoising-fluorescence"
        / "experiments"
        / "n2n_raw_unetv2_256_e300"
        / "May_19"
        / "unetv2_noise_train[1]_test[1]_four_crop_epochs300_bs2_lr0.0001"
        / "checkpoints"
        / "model_epoch300.pth"
    )
    print(f"Loading checkpoint: {ckpt_path}")

    model = UnetN2Nv2(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Gather files
    raw_files = sorted([f for f in test_mix_raw.iterdir() if f.suffix.lower() == '.png'])
    print(f"Found {len(raw_files)} test images")

    sample_names = []
    for raw_file in raw_files:
        sample_name = raw_file.stem
        sample_names.append(sample_name)
        sample_out = samples_dir / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)

        gt_file = test_mix_gt / raw_file.name

        # Load images
        noisy_img = Image.open(raw_file)
        gt_img = Image.open(gt_file)

        # Preprocess and infer
        noisy_t = preprocess(noisy_img).unsqueeze(0).to(device)
        with torch.no_grad():
            denoised_t = model(noisy_t)

        # Convert back to images
        noisy_arr = postprocess(noisy_t[0])
        denoised_arr = postprocess(denoised_t[0])
        gt_arr = postprocess(preprocess(gt_img))

        # Save
        Image.fromarray(noisy_arr, mode='L').save(sample_out / "before_network.png")
        Image.fromarray(denoised_arr, mode='L').save(sample_out / "after_network.png")
        Image.fromarray(gt_arr, mode='L').save(sample_out / "gt.png")

        print(f"  Saved {sample_name}")

    # Write manifest
    manifest = {
        "dataset": "FMD",
        "model": "n2n_raw_unetv2_256_e300",
        "sample_dir": "samples",
        "sample_count": len(sample_names),
        "sample_files": ["before_network.png", "after_network.png", "gt.png"],
        "gt": True,
        "notes": "FMD test_mix fluorescence microscopy denoising results."
    }
    with open(result_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print(f"\nDone. Generated {len(sample_names)} samples in {result_dir}")


if __name__ == "__main__":
    main()
