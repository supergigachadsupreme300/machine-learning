"""Configuration for Quy-model (landmark-based via MediaPipe)."""
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
ROOT_PROJ = PROJ_ROOT.parent

DATA_DIR = PROJ_ROOT / "data"
if not (DATA_DIR / "raw").exists() or not any((DATA_DIR / "raw").rglob("*.jpg")):
    if (ROOT_PROJ / "data").exists():
        DATA_DIR = ROOT_PROJ / "data"

DATA_ROOT = DATA_DIR / "raw"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

HAND_LANDMARKER = ROOT_PROJ / "machine_learning" / "ui" / "models" / "hand_landmarker.task"

IMG_SIZE = 64  # not used for landmarks, kept for compat
TEST_SIZE = 0.2
SEED = 42

MODEL_NAME = "random_forest"  # random_forest, svm, mlp, decision_tree
