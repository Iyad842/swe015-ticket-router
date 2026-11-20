"""Train the whole model ladder and evaluate it.

    python -m src.train_all                 # everything
    python -m src.train_all --skip-transformers   # only classical + FFN (fast, CPU)
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--skip-transformers", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    if not a.skip_data:
        from . import data
        print("\n=== 1/6 data ===")
        data.prepare()
    from . import baseline, embeddings, ffn
    print("\n=== 2/6 tf-idf + logistic regression ===")
    baseline.train()
    print("\n=== 3/6 own fastText embeddings ===")
    embeddings.train()
    print("\n=== 4/6 feed-forward network ===")
    ffn.train()
    if not a.skip_transformers:
        from . import transformer
        print("\n=== 5/6 mBERT teacher ===")
        transformer.train_teacher()
        print("\n=== 6/6 distilled student ===")
        transformer.distill_student()
    if not a.skip_eval:
        from . import evaluate
        print("\n=== evaluation ===")
        evaluate.main()
    print(f"\nall done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
