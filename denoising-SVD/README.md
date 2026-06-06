# Complex SVD Noise2Noise

This folder trains a Noise2Noise U-Net on paired complex SVD frames from
`F:\NOise_USdata\SVD_Noise`.

Default split:

- train: `M1`, `M2`, `M3`, `R1`, `R2`, `R3`, `R4`, `R5`
- test: `M4`, `R6`
- generalization: `P1`, `P2`, `P3`

Every split uses sorted frame pairs `0000, 0010, ..., 0490`. Each 512x512 frame is
cut into 256x256 patches with stride 128. Train patches additionally use
0/90/180/270 degree rotations.

Run a one-batch smoke check:

```powershell
python .\denoising-SVD\smoke_test.py
```

Start training:

```powershell
python .\denoising-SVD\train_n2n_complex.py --epochs 100 --batch-size 4 --cuda 0
```

After training, the script exports one network PDI map for each train/test
Mouse/Rat group under `network_pdi`. Each map uses
`sum_t(abs(network(full_svd_t)) ** 2)` from `F:\NOise_USdata\SVD_Noise_full`
and saves both the linear PDI `.npy` data and a 45 dB `.png` view.
