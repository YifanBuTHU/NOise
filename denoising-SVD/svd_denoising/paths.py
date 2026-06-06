from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = r"F:\NOise_USdata\SVD_Noise"
DEFAULT_FULL_DATA_ROOT = r"F:\NOise_USdata\SVD_Noise_full"


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
