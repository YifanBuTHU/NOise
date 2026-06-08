# NOise 数据与结果目录规范

本文档记录 `NOise` 工作区的数据根、结果根和命名规则。默认数据根为 `F:\NOise_data`，训练代码保留在 `E:\CodeField\codePython\NOise`。

## 总体结构

```text
F:\NOise_data\
  FMD\
  US\
    b_mode\
    even_odd\
    svd_noise\
    svd_noise_full\
  CT\
    ground_truth_test\
    observation_test\
    patient_ids_rand_test.csv
  MRC\
  CARE_2D\
    train_low_snr\
    train_high_snr\
    test_low_snr\
    test_high_snr\
  prepared\
    fluorescence_generalization_inputs\
    ct_noisy_generalization\
    svd_generalization_full\
  N2N_result\
    CARE_2D\
    CT\
    MRC\
    US\
```

`N2N_result` 只保留已有、非空、方便人工快速查看的泛化结果。当前没有可用的 FMD 图像结果，因此不保留空的 `N2N_result\FMD`。

## 命名规则

- 目录使用 `snake_case`，数据集根保留清晰的大写缩写，例如 `FMD`、`US`、`CT`、`MRC`、`CARE_2D`。
- 原始数据放在对应数据集根目录下；从原始数据生成、但还不是网络输出的中间输入放在 `prepared`。
- `N2N_result` 存放已经整理好的 PNG 泛化结果和 `manifest.json`。
- 每个数据集的 N2N 泛化结果统一使用 `n2n_raw_unetv2_256_e300` 作为模型结果目录名。
- 样本级目录放在 `samples` 下，文件名优先使用 `before_network.png`、`after_network.png`、`gt.png`。没有 GT 的数据集不强行创建 `gt.png`。
- 不再在 `N2N_result` 中保留组合对比图，尤其是 `comparison`、`comparision`、`pair`、`overview` 这类文件。

## 当前结果位置

```text
F:\NOise_data\N2N_result\
  CARE_2D\
    n2n_raw_unetv2_256_e300\
      samples\
      manifest.json
  CT\
    n2n_raw_unetv2_256_e300\
      samples\
      manifest.json
  MRC\
    n2n_raw_unetv2_256_e300\
      samples\
      manifest.json
  US\
    n2n_raw_unetv2_256_e300\
      samples\
      manifest.json
```

US complex SVD 的 `network_pdi` 结果不属于 `N2N_result`，保留在对应训练 run 的代码侧目录：

```text
E:\CodeField\codePython\NOise\denoising-SVD\experiments\complex_n2n\20260528-215406\network_pdi
```

## 入口脚本默认路径

- `denoising-fluorescence/denoising/paths.py`
  - FMD 数据：`F:\NOise_data\FMD`
- `denoising-SVD/svd_denoising/paths.py`
  - 数据根：`F:\NOise_data`
  - SVD N2N 训练数据：`F:\NOise_data\US\svd_noise`
  - full SVD PDI 数据：`F:\NOise_data\US\svd_noise_full`
  - SVD 外部泛化预处理数据：`F:\NOise_data\prepared\svd_generalization_full`
  - N2N 结果根：`F:\NOise_data\N2N_result`
- `denoising-SVD/prepare_external_generalization_data.py`
  - 读取 `MRC` 和 `CARE_2D`
  - 默认输出到 `F:\NOise_data\prepared\svd_generalization_full`
- `F:\NOise_data\prepared\ct_noisy_generalization`
  - 只保留 `clean` 和 `noisy_gaussian_sigma003`
- `denoising-SVD/train_n2n_complex.py`
  - checkpoint、metrics、训练预览和 `network_pdi` 保留在项目 `experiments`
- `denoising-SVD/generate_final_pdi.py`、`supplement_network_pdi.py`、`generate_generalization_pdi.py`
  - 默认输出到对应 run 的 `network_pdi`
- `analyze_ct.py`
  - CT 分析图和报告输出到 `F:\NOise_data\N2N_result\CT\analysis\ct_analysis`

## 维护原则

- 不把新结果直接写进原始数据目录。
- 训练过程产物留在各项目 `experiments`；整理后的泛化查看结果放在 `N2N_result`。
- 移动或改名数据目录前，先检查目标目录是否存在且非空；有冲突时停止，不覆盖。
- 新脚本优先复用现有 `paths.py` 中的数据根和结果根常量。
