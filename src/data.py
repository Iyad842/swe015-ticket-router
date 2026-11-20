"""Download CLINC150 (oos_plus), add hand-written appliance examples,
normalize everything and write train/val/test CSV files.

Run:  python -m src.data
"""
from __future__ import annotations

import json
import random
import urllib.request
from collections import Counter

import pandas as pd

from .config import DATA_DIR, MODELS_DIR, OOS_LABEL, PROCESSED_DIR, SEED
from .routing import domain_of
from .text_utils import Normalizer, tokenize, tr_lower

CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"


def load_clinc() -> dict[str, list[tuple[str, str]]]:
    """Return {'train': [(text, intent)], 'val': [...], 'test': [...]}."""
    try:
        print(f"Downloading CLINC150 (oos_plus) from {CLINC_URL}")
        with urllib.request.urlopen(CLINC_URL, timeout=60) as r:
            raw = json.loads(r.read().decode("utf-8"))
        out = {}
        for split in ("train", "val", "test"):
            out[split] = [tuple(x) for x in raw[split]] + [tuple(x) for x in raw[f"oos_{split}"]]
        return out
    except Exception as e:  # fall back to the Hugging Face copy
        print(f"GitHub download failed ({e}); trying Hugging Face datasets ...")
        from datasets import load_dataset
        ds = load_dataset("clinc_oos", "plus")
        names = ds["train"].features["intent"].names
        m = {"train": "train", "val": "validation", "test": "test"}
        return {s: [(r["text"], names[r["intent"]]) for r in ds[h]] for s, h in m.items()}


def load_appliance() -> dict[str, list[tuple[str, str]]]:
    data = json.loads((DATA_DIR / "appliance_extra.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    out = {"train": [], "val": [], "test": []}
    for intent, sents in data.items():
        if intent.startswith("_"):
            continue
        sents = sents[:]
        rng.shuffle(sents)
        out["train"] += [(s, intent) for s in sents[:10]]
        out["val"] += [(s, intent) for s in sents[10:12]]
        out["test"] += [(s, intent) for s in sents[12:16]]
    return out


def build_vocab(texts: list[str], min_count: int = 2) -> set[str]:
    """Known words = tokens seen >= min_count times in the *training* data.
    (Typos are rare, so they do not enter the vocabulary.)"""
    c = Counter(t for s in texts for t in tokenize(tr_lower(s)))
    return {w for w, n in c.items() if n >= min_count}


def prepare() -> None:
    clinc, appl = load_clinc(), load_appliance()
    splits = {s: clinc[s] + appl[s] for s in ("train", "val", "test")}

    vocab = build_vocab([t for t, _ in splits["train"]])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "vocab.json").write_text(json.dumps(sorted(vocab), ensure_ascii=False), encoding="utf-8")
    norm = Normalizer(vocab)

    labels = sorted({i for _, i in splits["train"]})
    assert OOS_LABEL in labels
    (MODELS_DIR / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        recs = []
        for text, intent in rows:
            n = norm(text)
            recs.append({"text": text, "clean": n["clean"], "lower": n["lower"],
                         "intent": intent, "domain": domain_of(intent), "lang": n["lang"]})
        df = pd.DataFrame(recs)
        df.to_csv(PROCESSED_DIR / f"{split}.csv", index=False)
        print(f"{split:5s}: {len(df):6d} rows | oos={int((df.intent == OOS_LABEL).sum())} "
              f"| appliance={int((df.domain == 'appliance').sum())} | tr={int((df.lang == 'tr').sum())}")
    unmapped = [l for l in labels if domain_of(l) == "general"]
    if unmapped:
        print("WARNING: intents without a domain (routed to General Support):", unmapped)
    print(f"labels={len(labels)} vocab={len(vocab)} -> {PROCESSED_DIR}")


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{split}.csv", keep_default_na=False)
    return df


def load_labels() -> list[str]:
    return json.loads((MODELS_DIR / "labels.json").read_text())


if __name__ == "__main__":
    prepare()
