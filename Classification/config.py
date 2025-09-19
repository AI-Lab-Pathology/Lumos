import os
import torch


IMAGE_ROOT = r"\train\image"
TEXT_ROOT  = r"\train\txt"
VAL_IMAGE_ROOT = r"\val\image"
VAL_TEXT_ROOT  = r"\val\txt"

BERT_MODEL_PATH = r"BioBERT"  
SAVE_PATH = "checkpoints/cross_modal_classifier.pt"
BASE_VIS_DIR = "visualization_report"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(BASE_VIS_DIR, exist_ok=True)


NUM_CLASSES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-5


WARMUP_RATIO = 0.1


SEED = 42
