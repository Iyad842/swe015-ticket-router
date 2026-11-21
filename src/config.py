"""Central configuration: paths, model names and hyper-parameters."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"          # git-ignored; downloaded from Hugging Face Hub
RESULTS_DIR = ROOT / "results"

SEED = 42

# ---- Transformers ---------------------------------------------------------
# Teacher: multilingual BERT (understands Turkish and English).
TEACHER_NAME = "bert-base-multilingual-cased"
# Student: 6-layer DistilBERT that shares the same WordPiece tokenizer,
# so the teacher's logits can be distilled directly.
STUDENT_NAME = "distilbert-base-multilingual-cased"
MAX_LEN = 64

TEACHER_EPOCHS = 4
TEACHER_LR = 4e-5
STUDENT_EPOCHS = 5
STUDENT_LR = 5e-5
BATCH_SIZE = 32
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5          # weight of the soft (teacher) loss

# ---- Own fastText-style embeddings + feed-forward network ----------------
EMB_DIM = 100
EMB_BUCKETS = 100_000
EMB_EPOCHS = 5
FFN_HIDDEN = 512
FFN_EPOCHS = 40
FFN_LR = 0.1

# ---- Hugging Face Hub repo that stores the trained weights (R7) ----------
HF_REPO_ID = os.environ.get("ROUTER_HF_REPO", "IYAD842/swe015-ticket-router")

OOS_LABEL = "oos"
