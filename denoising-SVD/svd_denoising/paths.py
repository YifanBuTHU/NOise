from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = r"F:\NOise_data"
DEFAULT_DATA_ROOT = rf"{DATA_ROOT}\US\svd_noise"
DEFAULT_FULL_DATA_ROOT = rf"{DATA_ROOT}\US\svd_noise_full"
DEFAULT_GENERALIZATION_FULL_ROOT = rf"{DATA_ROOT}\prepared\svd_generalization_full"
DEFAULT_RESULT_ROOT = rf"{DATA_ROOT}\N2N_result"


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative_path(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path
