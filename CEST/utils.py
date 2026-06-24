import os
from pathlib import Path


def create_directories(exp_path, slice_index=None):
    if slice_index is not None:
        log_dir = str(Path(exp_path) / f"slice_{slice_index}" / "logs")
        model_dir = str(Path(exp_path) / f"slice_{slice_index}" / "models")
    else:
        log_dir = str(Path(exp_path) / "logs")
        model_dir = str(Path(exp_path) / "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir
