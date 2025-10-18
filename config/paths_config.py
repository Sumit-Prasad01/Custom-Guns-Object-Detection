import torch

# Paths
ROOT_PATH = "artifacts/raw/"
MODEL_SAVE_PATH = "artifacts/models/"


# Device Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"