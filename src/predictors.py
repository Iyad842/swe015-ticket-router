"""Common interface for the five models of the ladder.

Each predictor takes a list of normalized messages (dicts produced by
text_utils.Normalizer, needing the keys 'clean' and 'lower') and returns a
probability matrix whose columns follow models/labels.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, OOS_LABEL, RESULTS_DIR


def _size_mb(*paths: Path) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*")
                         if f.is_file() and f.suffix in {".safetensors", ".bin"})
        elif p.exists():
            total += p.stat().st_size
    return total / 1e6


def load_labels() -> list[str]:
    return json.loads((MODELS_DIR / "labels.json").read_text())


class Predictor:
    key = name = precision = ""
    files: list[Path] = []

    def predict_proba(self, norms: list[dict]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def size_mb(self) -> float:
        return _size_mb(*self.files)


class BaselinePredictor(Predictor):
    key, name, precision = "baseline", "TF-IDF + Logistic Regression", "FP64"

    def __init__(self, labels):
        import joblib
        from .baseline import PATH
        self.files = [PATH]
        self.model = joblib.load(PATH)
        classes = list(self.model.classes_)
        self.order = [classes.index(l) for l in labels]

    def predict_proba(self, norms):
        return self.model.predict_proba([n["lower"] for n in norms])[:, self.order]


class FFNPredictor(Predictor):
    key, name, precision = "ffn", "FFN on own fastText embeddings", "FP32"

    def __init__(self, labels):
        import torch
        from .embeddings import PATH as EMB_PATH, FastTextVectors
        from .ffn import PATH, build
        self.torch = torch
        self.files = [EMB_PATH, PATH]
        self.vec = FastTextVectors.load(EMB_PATH)
        ck = torch.load(PATH, map_location="cpu", weights_only=False)
        assert ck["labels"] == labels, "FFN was trained with a different label set"
        self.model = build(ck["dim"], ck["hidden"], len(labels))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.mean, self.std = ck["mean"], ck["std"]

    def predict_proba(self, norms):
        X = np.stack([self.vec.sentence_vector(n["lower"]) for n in norms])
        X = (X - self.mean) / self.std
        with self.torch.no_grad():
            logits = self.model(self.torch.tensor(X, dtype=self.torch.float32))
        return self.torch.softmax(logits, -1).numpy()


class TransformerPredictor(Predictor):
    def __init__(self, key, name, path, int8=False, device="cpu"):
        import torch
        from .transformer import INT8_PATH, load_classifier
        self.torch = torch
        self.key, self.name = key, name
        self.model, self.tok, self.precision, self.device = load_classifier(path, int8, device)
        self.files = [INT8_PATH] if int8 else [path]

    def to(self, device: str):
        if not self.precision.startswith("INT8"):
            self.model.to(device)
            self.device = device
        return self

    def predict_proba(self, norms, bs: int = 64):
        from .transformer import encode
        texts = [n["clean"] for n in norms]
        out = []
        with self.torch.no_grad():
            for b in range(0, len(texts), bs):
                logits = self.model(**encode(self.tok, texts[b:b + bs], self.device)).logits
                out.append(self.torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out)


# key -> (display name, factory)
def _factories(labels, device):
    from .transformer import STUDENT_DIR, TEACHER_DIR
    return {
        "baseline": lambda: BaselinePredictor(labels),
        "ffn": lambda: FFNPredictor(labels),
        "teacher": lambda: TransformerPredictor("teacher", "mBERT fine-tuned (teacher)", TEACHER_DIR, device=device),
        "student": lambda: TransformerPredictor("student", "DistilmBERT distilled (student)", STUDENT_DIR, device=device),
        "student_int8": lambda: TransformerPredictor("student_int8", "DistilmBERT distilled + INT8 quantized",
                                                     STUDENT_DIR, int8=True),
    }


ALL_KEYS = ["baseline", "ffn", "teacher", "student", "student_int8"]
PRIMARY_KEY = "student_int8"      # the model the application uses (R5)


def load_predictors(keys=None, device: str = "cpu") -> dict[str, Predictor]:
    labels = load_labels()
    fac = _factories(labels, device)
    out = {}
    for k in keys or ALL_KEYS:
        try:
            out[k] = fac[k]()
            print(f"loaded {k}: {out[k].name} [{out[k].precision}]")
        except Exception as e:  # keep the app usable if one model is missing
            print(f"could not load {k}: {type(e).__name__}: {e}")
    return out


def load_thresholds() -> dict[str, float]:
    for p in (RESULTS_DIR / "thresholds.json", MODELS_DIR / "thresholds.json"):
        if p.exists():
            return json.loads(p.read_text())
    return {}


def decide(probs: np.ndarray, labels: list[str], tau: float) -> tuple[str, float, str]:
    """Return (final_label, confidence, raw_top_label). Low confidence -> 'oos'."""
    i = int(np.argmax(probs))
    conf, top = float(probs[i]), labels[i]
    final = OOS_LABEL if (top == OOS_LABEL or conf < tau) else top
    return final, conf, top


def set_threads() -> None:
    n = os.environ.get("ROUTER_THREADS")
    if n:
        import torch
        torch.set_num_threads(int(n))
