"""Own fastText-style word embeddings (skip-gram with negative sampling,
sub-word character n-grams), trained on the training messages.

A word vector is the mean of the vectors of its character 3-5-grams
(plus a whole-word vector for in-vocabulary words), hashed into a fixed
number of buckets. Sub-words make the embeddings robust to typos and to
Turkish suffixes ("makinem", "makinesinin", ...).

Run:  python -m src.embeddings
"""
from __future__ import annotations

import time
import zlib
from collections import Counter

import numpy as np

from .config import EMB_BUCKETS, EMB_DIM, EMB_EPOCHS, MODELS_DIR, SEED

PATH = MODELS_DIR / "fasttext_own.pt"
MINN, MAXN, WINDOW, NEGATIVES = 3, 5, 3, 5


def subword_ids(word: str, n_buckets: int, in_vocab: bool) -> list[int]:
    w = f"<{word}>"
    grams = [w[i:i + n] for n in range(MINN, MAXN + 1) for i in range(len(w) - n + 1)]
    ids = [zlib.crc32(g.encode("utf-8")) % n_buckets for g in grams]
    if in_vocab:  # whole-word vector, separate key space
        ids.append(zlib.crc32(("w::" + word).encode("utf-8")) % n_buckets)
    return ids or [zlib.crc32(w.encode("utf-8")) % n_buckets]


# --------------------------------------------------------------------------
# Inference (numpy only)
# --------------------------------------------------------------------------
class FastTextVectors:
    def __init__(self, weight: np.ndarray, vocab: list[str], n_buckets: int):
        self.weight, self.vocab, self.n_buckets = weight, set(vocab), n_buckets
        self.dim = weight.shape[1]
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, path=PATH) -> "FastTextVectors":
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return cls(ck["weight"].float().numpy(), ck["vocab"], ck["n_buckets"])

    def word_vector(self, word: str) -> np.ndarray:
        v = self._cache.get(word)
        if v is None:
            v = self.weight[subword_ids(word, self.n_buckets, word in self.vocab)].mean(0)
            self._cache[word] = v
        return v

    def sentence_vector(self, lower_text: str) -> np.ndarray:
        toks = lower_text.split()
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean([self.word_vector(t) for t in toks], axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# Training (PyTorch)
# --------------------------------------------------------------------------
def train() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .data import load_split

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    sents = [s.split() for s in load_split("train")["lower"]]
    counts = Counter(w for s in sents for w in s)
    vocab = sorted(counts)
    w2i = {w: i for i, w in enumerate(vocab)}
    V, B = len(vocab), EMB_BUCKETS

    # padded matrix of sub-word ids per vocabulary word
    subs = [subword_ids(w, B, True) for w in vocab]
    L = max(len(s) for s in subs)
    sub_mat = np.full((V, L), B, dtype=np.int64)          # B = padding index
    for i, s in enumerate(subs):
        sub_mat[i, :len(s)] = s
    sub_len = np.array([len(s) for s in subs], dtype=np.float32)

    # (center, context) pairs inside a +-WINDOW window
    centers, contexts = [], []
    for s in sents:
        ids = [w2i[w] for w in s]
        for i, c in enumerate(ids):
            for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
                if j != i:
                    centers.append(c)
                    contexts.append(ids[j])
    centers, contexts = np.array(centers), np.array(contexts)
    print(f"vocab={V} pairs={len(centers)} max_subwords={L}")

    noise = torch.tensor([counts[w] for w in vocab], dtype=torch.float) ** 0.75
    noise /= noise.sum()
    sub_mat_t, sub_len_t = torch.from_numpy(sub_mat), torch.from_numpy(sub_len)

    class SkipGram(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Embedding(B + 1, EMB_DIM, padding_idx=B, sparse=True)
            self.out = nn.Embedding(V, EMB_DIM, sparse=True)
            nn.init.uniform_(self.inp.weight, -0.5 / EMB_DIM, 0.5 / EMB_DIM)
            with torch.no_grad():
                self.inp.weight[B].zero_()
            nn.init.zeros_(self.out.weight)

        def forward(self, c, ctx, neg):
            center = self.inp(sub_mat_t[c]).sum(1) / sub_len_t[c].unsqueeze(1)   # (b, d)
            pos = (center * self.out(ctx)).sum(-1)                               # (b,)
            negs = torch.bmm(self.out(neg), center.unsqueeze(2)).squeeze(2)     # (b, k)
            return -(F.logsigmoid(pos) + F.logsigmoid(-negs).sum(1)).mean()

    model = SkipGram()
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    bs = 2048
    for ep in range(EMB_EPOCHS):
        t0, tot, nb = time.time(), 0.0, 0
        perm = rng.permutation(len(centers))
        for b in range(0, len(perm), bs):
            idx = perm[b:b + bs]
            c = torch.from_numpy(centers[idx])
            ctx = torch.from_numpy(contexts[idx])
            neg = torch.multinomial(noise, len(idx) * NEGATIVES, replacement=True).view(len(idx), NEGATIVES)
            loss = model(c, ctx, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        print(f"epoch {ep + 1}/{EMB_EPOCHS} loss={tot / nb:.4f} ({time.time() - t0:.0f}s)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": model.inp.weight.detach()[:B].clone(), "vocab": vocab,
                "n_buckets": B, "dim": EMB_DIM}, PATH)
    print("saved", PATH)

    # quick sanity check: nearest neighbours
    vec = FastTextVectors.load(PATH)
    for q in ["dishwasher", "bulaşık", "transfer", "flight"]:
        if q in vec.vocab:
            qv = vec.word_vector(q)
            sims = []
            for w in sorted(vec.vocab):
                wv = vec.word_vector(w)
                sims.append((float(qv @ wv / (np.linalg.norm(qv) * np.linalg.norm(wv) + 1e-9)), w))
            print(q, "->", [w for _, w in sorted(sims, reverse=True)[1:6]])


if __name__ == "__main__":
    train()
