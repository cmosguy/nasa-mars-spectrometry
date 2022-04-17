from pathlib import Path
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42  # For reproducibility

DATA_PATH = Path.cwd() / "data/final/public/"