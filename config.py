
from pathlib import Path
import torch

# Project Paths
ROOT_DIR = Path(__file__).resolve().parent

# Your folders are directly inside data/
DATA_DIR = ROOT_DIR / "data" / "Brain_Tumor_MRI_Dataset"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "outputs"

PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Training Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2

NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
NUM_CLASSES = 4
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Checkpoints
BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL = CHECKPOINT_DIR / "last_model.pth"

PATIENCE = 10
SAVE_BEST_ONLY = True
USE_AMP = torch.cuda.is_available()
