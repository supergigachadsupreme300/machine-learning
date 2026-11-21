"""Configuration for Quy-model gesture classifier (landmark-based)."""
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
ROOT_PROJ = PROJ_ROOT.parent

# Prefer local data folder, fall back to repository-root data folder
DATA_DIR = PROJ_ROOT / "data"
if not (DATA_DIR / "raw").exists() or not any((DATA_DIR / "raw").rglob("*.csv")):
    if (ROOT_PROJ / "data").exists():
        DATA_DIR = ROOT_PROJ / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
SPLIT_DATA_DIR = DATA_DIR / "split"
TRAIN_DIR = SPLIT_DATA_DIR / "train"
VAL_DIR = SPLIT_DATA_DIR / "val"
TEST_DIR = SPLIT_DATA_DIR / "test"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

# Landmark features: 21 MediaPipe hand landmarks * (x, y, z) = 63
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

# Training
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42

# Model selection: "random_forest", "svm", "knn", "gradient_boosting",
# "mlp", "logistic", "torch_mlp"
MODEL_NAME = "random_forest"
CV_FOLDS = 5

# PyTorch MLP (only used when MODEL_NAME == "torch_mlp")
TORCH_EPOCHS = 50
TORCH_BATCH_SIZE = 64
TORCH_LR = 1e-3
TORCH_HIDDEN = [128, 64]

DEVICE = "auto"  # auto/cpu/cuda
