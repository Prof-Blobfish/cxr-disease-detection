from pathlib import Path

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 1 - TRAIN_SPLIT - VAL_SPLIT

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

DATASET_DIR = "/Volumes/Secratary/Datasets/NIH Chest X-Rays"
# DATASET_DIR = "F:/Datasets/NIH_Chest_X-Rays"