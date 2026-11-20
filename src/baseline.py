"""Model 1 - classical baseline: tf-idf (word 1-2 grams + char 3-5 grams)
+ multinomial logistic regression.   Run: python -m src.baseline
"""
from __future__ import annotations

import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import MODELS_DIR
from .data import load_split

PATH = MODELS_DIR / "baseline_tfidf_lr.joblib"


def build() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        # character n-grams make the baseline robust to typos and Turkish suffixes
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(C=10.0, max_iter=3000))])


def train() -> None:
    tr, va = load_split("train"), load_split("val")
    t0 = time.time()
    model = build().fit(tr["lower"], tr["intent"])
    print(f"baseline trained in {time.time() - t0:.1f}s | val acc = {model.score(va['lower'], va['intent']):.4f}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATH)
    print("saved", PATH)


if __name__ == "__main__":
    train()
