"""Models 3-5 - Transformers (area C) and efficient models (area D).

3. Teacher : bert-base-multilingual-cased fine-tuned with a classification head.
4. Student : distilbert-base-multilingual-cased trained by knowledge
             distillation from the teacher (soft targets, temperature T).
5. Student INT8 : the student after post-training dynamic quantization
             (Linear weights -> int8; embeddings -> int8 if supported).

Run:  python -m src.transformer teacher     (GPU recommended, e.g. Colab T4)
      python -m src.transformer student
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (BATCH_SIZE, KD_ALPHA, KD_TEMPERATURE, MAX_LEN, MODELS_DIR, SEED,
                     STUDENT_EPOCHS, STUDENT_LR, STUDENT_NAME, TEACHER_EPOCHS, TEACHER_LR,
                     TEACHER_NAME)

TEACHER_DIR = MODELS_DIR / "teacher"
STUDENT_DIR = MODELS_DIR / "student"
INT8_PATH = MODELS_DIR / "student_int8_state_dict.pt"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(tok, texts, device=None):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    return enc.to(device) if device is not None else enc


@torch.no_grad()
def predict_logits(model, tok, texts, device, bs: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    texts = list(texts)
    for b in range(0, len(texts), bs):
        out.append(model(**encode(tok, texts[b:b + bs], device)).logits.float().cpu())
    return torch.cat(out)


def linear_warmup_decay(opt, total_steps: int, warmup_frac: float = 0.1):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def new_classifier(name: str, labels: list[str]):
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)})


def fine_tune(model, tok, tr_texts, ytr, va_texts, yva, epochs, lr, out_dir,
              teacher_logits: torch.Tensor | None = None) -> float:
    """Generic training loop. If teacher_logits is given, the loss is
    alpha * T^2 * KL(student_T || teacher_T) + (1 - alpha) * CE(student, y)."""
    device = train_device()
    model.to(device)
    rng = np.random.default_rng(SEED)
    n = len(ytr)
    steps = epochs * math.ceil(n / BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = linear_warmup_decay(opt, steps)
    best, log = -1.0, []
    T = KD_TEMPERATURE
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = rng.permutation(n)
        for b in range(0, n, BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            logits = model(**encode(tok, tr_texts[idx], device)).logits
            y = torch.as_tensor(ytr[idx], device=device)
            loss = F.cross_entropy(logits, y)
            if teacher_logits is not None:
                t = teacher_logits[idx].to(device)
                kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t / T, -1),
                              reduction="batchmean") * T * T
                loss = KD_ALPHA * kd + (1 - KD_ALPHA) * loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        va_acc = float((predict_logits(model, tok, va_texts, device).argmax(1).numpy() == yva).mean())
        log.append({"epoch": ep + 1, "train_loss": tot / n, "val_acc": va_acc})
        print(f"epoch {ep + 1}/{epochs} loss={tot / n:.4f} val_acc={va_acc:.4f} ({time.time() - t0:.0f}s)")
        if va_acc > best:
            best = va_acc
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"best val acc={best:.4f} -> {out_dir}")
    return best


def _data():
    from .data import load_labels, load_split
    labels = load_labels()
    l2i = {l: i for i, l in enumerate(labels)}
    tr, va = load_split("train"), load_split("val")
    tr_x = np.array(tr["clean"].tolist(), dtype=object)
    va_x = np.array(va["clean"].tolist(), dtype=object)
    ytr = np.array([l2i[l] for l in tr["intent"]])
    yva = np.array([l2i[l] for l in va["intent"]])
    return labels, tr_x, ytr, va_x, yva


def train_teacher() -> None:
    set_seed()
    labels, tr_x, ytr, va_x, yva = _data()
    tok = AutoTokenizer.from_pretrained(TEACHER_NAME)
    model = new_classifier(TEACHER_NAME, labels)
    print(f"teacher {TEACHER_NAME} on {train_device()} | train={len(ytr)} labels={len(labels)}")
    fine_tune(model, tok, tr_x, ytr, va_x, yva, TEACHER_EPOCHS, TEACHER_LR, TEACHER_DIR)


def distill_student() -> None:
    set_seed()
    labels, tr_x, ytr, va_x, yva = _data()
    device = train_device()
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_DIR).to(device)
    tok_t = AutoTokenizer.from_pretrained(TEACHER_DIR)
    print("computing teacher soft targets on the training set ...")
    t_logits = predict_logits(teacher, tok_t, tr_x, device)
    print(f"teacher train acc={float((t_logits.argmax(1).numpy() == ytr).mean()):.4f}")
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(STUDENT_NAME)
    student = new_classifier(STUDENT_NAME, labels)
    print(f"student {STUDENT_NAME} (KD: T={KD_TEMPERATURE}, alpha={KD_ALPHA})")
    fine_tune(student, tok, tr_x, ytr, va_x, yva, STUDENT_EPOCHS, STUDENT_LR, STUDENT_DIR,
              teacher_logits=t_logits)


# --------------------------------------------------------------------------
# Quantization (area D)
# --------------------------------------------------------------------------
def pick_quant_engine() -> str:
    engines = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):     # qnnpack = ARM / Apple Silicon
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    return torch.backends.quantized.engine


def quantize_int8(model, try_embeddings: bool = True):
    """Post-training dynamic quantization on CPU.
    Linear layers: int8 weights, activations quantized on the fly.
    Embeddings  : int8 weight-only (if this torch version supports it)."""
    from torch.ao.quantization import (default_dynamic_qconfig, float_qparams_weight_only_qconfig,
                                       quantize_dynamic)
    model = model.cpu().eval()
    engine = pick_quant_engine()
    dummy = {"input_ids": torch.tensor([[101, 10111, 102]]),
             "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    if try_embeddings:
        try:
            q = quantize_dynamic(copy.deepcopy(model),
                                 {nn.Linear: default_dynamic_qconfig,
                                  nn.Embedding: float_qparams_weight_only_qconfig})
            with torch.no_grad():
                q(**dummy)
            left_fp32 = sum(type(m) is nn.Embedding for m in q.modules())
            if left_fp32 == 0:
                return q, f"INT8 (dynamic Linear + weight-only Embedding, {engine})"
            print(f"[quantize] {left_fp32} embedding layers stayed FP32")
            return q, f"INT8 (dynamic, Linear layers; {left_fp32} embeddings FP32, {engine})"
        except Exception as e:
            print(f"[quantize] embedding quantization not supported here ({type(e).__name__}); "
                  "falling back to Linear-only")
    q = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        q(**dummy)
    return q, f"INT8 (dynamic, Linear layers; embeddings FP32, {engine})"


def load_classifier(path, int8: bool = False, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    if int8:
        model, precision = quantize_int8(model)
        torch.save(model.state_dict(), INT8_PATH)    # only to measure the on-disk size
        return model, tok, precision, "cpu"
    return model.to(device), tok, "FP32", device


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("teacher", "all"):
        train_teacher()
    if what in ("student", "all"):
        distill_student()
