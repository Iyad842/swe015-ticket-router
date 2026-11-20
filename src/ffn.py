"""Model 2 - feed-forward neural network on averaged fastText embeddings,
trained with mini-batch SGD (momentum) and backpropagation.
Architecture: 100-d sentence vector -> Linear(512) -> ReLU -> Dropout(0.3) -> Linear(n_classes)

Run:  python -m src.ffn
"""
from __future__ import annotations

import json
import time

import numpy as np

from .config import FFN_EPOCHS, FFN_HIDDEN, FFN_LR, MODELS_DIR, SEED
from .embeddings import FastTextVectors

PATH = MODELS_DIR / "ffn.pt"


def build(dim: int, hidden: int, n_classes: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(hidden, n_classes))


def featurize(vec: FastTextVectors, texts) -> np.ndarray:
    return np.stack([vec.sentence_vector(t) for t in texts])


def train() -> None:
    import torch
    import torch.nn.functional as F

    from .data import load_labels, load_split

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    labels = load_labels()
    l2i = {l: i for i, l in enumerate(labels)}
    tr, va = load_split("train"), load_split("val")
    vec = FastTextVectors.load()

    Xtr, Xva = featurize(vec, tr["lower"]), featurize(vec, va["lower"])
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6           # standardize features
    Xtr, Xva = (Xtr - mean) / std, (Xva - mean) / std
    ytr = np.array([l2i[l] for l in tr["intent"]])
    yva = np.array([l2i[l] for l in va["intent"]])
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t = torch.tensor(Xva)

    model = build(Xtr.shape[1], FFN_HIDDEN, len(labels))
    opt = torch.optim.SGD(model.parameters(), lr=FFN_LR, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FFN_EPOCHS)
    best_acc, best_state, log = -1.0, None, []
    for ep in range(FFN_EPOCHS):
        model.train()
        t0, tot = time.time(), 0.0
        perm = rng.permutation(len(ytr))
        for b in range(0, len(perm), 64):
            idx = torch.from_numpy(perm[b:b + 64])
            loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()          # backpropagation
            opt.step()               # SGD update
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xva_t).argmax(1).numpy() == yva).mean())
        log.append({"epoch": ep + 1, "train_loss": tot / len(ytr), "val_acc": acc})
        print(f"epoch {ep + 1:2d} loss={tot / len(ytr):.4f} val_acc={acc:.4f} ({time.time() - t0:.1f}s)")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save({"state_dict": best_state, "mean": mean, "std": std, "dim": Xtr.shape[1],
                "hidden": FFN_HIDDEN, "labels": labels}, PATH)
    (MODELS_DIR / "ffn_train_log.json").write_text(json.dumps(log, indent=1))
    print(f"best val acc={best_acc:.4f} saved {PATH}")


if __name__ == "__main__":
    train()
