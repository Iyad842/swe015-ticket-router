"""Evaluate every model of the ladder on the SAME held-out test set (R3)
and build the results table (R4): accuracy, out-of-scope recall,
model size (MB) and CPU latency (ms, mean over >= 100 single-message runs).

The out-of-scope threshold of every model is chosen on the validation set
only (never on the test set).

Run:  python -m src.evaluate
Outputs: results/results.json, results/results.md, results/thresholds.json,
         results/test_predictions.csv, results/failures_student_int8.csv
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd

from .config import MODELS_DIR, OOS_LABEL, RESULTS_DIR, SEED
from .data import load_split
from .predictors import ALL_KEYS, load_labels, load_predictors
from .text_utils import Normalizer

N_LATENCY = 200
N_WARMUP = 20
GRID = np.round(np.arange(0.0, 0.96, 0.05), 2)


def hardware_info() -> str:
    import torch
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        elif platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        elif platform.system() == "Windows":
            cpu = platform.processor()
    except Exception:
        pass
    return (f"{cpu}, {os.cpu_count()} logical cores, {platform.system()} {platform.release()}, "
            f"Python {platform.python_version()}, torch {torch.__version__} "
            f"({torch.get_num_threads()} threads), CPU only, batch size 1")


def final_predictions(probs, labels, tau):
    arg = probs.argmax(1)
    conf = probs.max(1)
    pred = np.array(labels, dtype=object)[arg]
    pred = np.where((conf < tau) | (pred == OOS_LABEL), OOS_LABEL, pred)
    return pred, conf


def metrics(pred, gold) -> dict:
    gold = np.asarray(gold, dtype=object)
    oos = gold == OOS_LABEL
    pred_oos = pred == OOS_LABEL
    return {
        "accuracy": float((pred == gold).mean()),
        "in_scope_accuracy": float((pred[~oos] == gold[~oos]).mean()),
        "oos_recall": float(pred_oos[oos].mean()),
        "oos_precision": float((oos & pred_oos).sum() / max(1, pred_oos.sum())),
    }


def choose_threshold(probs, labels, gold) -> float:
    best_tau, best_acc = 0.0, -1.0
    for tau in GRID:
        acc = metrics(final_predictions(probs, labels, tau)[0], gold)["accuracy"]
        if acc > best_acc + 1e-9:
            best_tau, best_acc = float(tau), acc
    return best_tau


def measure_latency(pred, normalizer, messages) -> tuple[float, float]:
    for m in messages[:N_WARMUP]:
        pred.predict_proba([normalizer(m)])
    times = []
    for m in messages:
        t0 = time.perf_counter()
        pred.predict_proba([normalizer(m)])        # full path: normalize + model
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main() -> None:
    import torch
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    va, te = load_split("val"), load_split("test")
    va_norms = va[["clean", "lower"]].to_dict("records")
    te_norms = te[["clean", "lower"]].to_dict("records")
    normalizer = Normalizer.from_file(MODELS_DIR / "vocab.json")
    lat_msgs = te["text"].sample(N_LATENCY, random_state=SEED).tolist()
    gpu = "cuda" if torch.cuda.is_available() else "cpu"

    preds = load_predictors(ALL_KEYS, device=gpu)   # accuracy pass may use the GPU
    rows, thresholds = [], {}
    pred_table = te[["text", "intent", "domain", "lang"]].copy()
    for key, p in preds.items():
        t0 = time.time()
        pv = p.predict_proba(va_norms)
        tau = choose_threshold(pv, labels, va["intent"].values)
        pt = p.predict_proba(te_norms)
        pred, conf = final_predictions(pt, labels, tau)
        m = metrics(pred, te["intent"].values)
        pred_table[f"{key}_pred"] = pred
        pred_table[f"{key}_conf"] = np.round(conf, 4)
        if hasattr(p, "to"):
            p.to("cpu")                                # latency is always measured on CPU
        lat_mean, lat_p95 = measure_latency(p, normalizer, lat_msgs)
        thresholds[key] = tau
        row = {"key": key, "model": p.name, "precision": p.precision, **m,
               "threshold": tau, "size_mb": round(p.size_mb(), 1),
               "latency_ms_mean": round(lat_mean, 2), "latency_ms_p95": round(lat_p95, 2)}
        rows.append(row)
        print(f"{key:13s} acc={m['accuracy']:.4f} in-scope={m['in_scope_accuracy']:.4f} "
              f"oos_recall={m['oos_recall']:.4f} tau={tau} size={row['size_mb']}MB "
              f"lat={lat_mean:.2f}ms ({time.time() - t0:.0f}s)")

    # per-language and appliance-domain accuracy (useful for the conclusion slide)
    extra = {}
    for key in preds:
        col = pred_table[f"{key}_pred"]
        extra[key] = {
            "appliance_accuracy": float((col[te.domain == "appliance"] == te.intent[te.domain == "appliance"]).mean()),
            "turkish_accuracy": float((col[te.lang == "tr"] == te.intent[te.lang == "tr"]).mean())
            if (te.lang == "tr").any() else None,
        }

    results = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware_info(),
        "n_val": len(va), "n_test": len(te),
        "n_test_oos": int((te.intent == OOS_LABEL).sum()),
        "n_test_appliance": int((te.domain == "appliance").sum()),
        "n_latency_runs": N_LATENCY,
        "models": rows, "subsets": extra,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (RESULTS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))  # shipped with the weights
    pred_table.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    if "student_int8_pred" in pred_table:
        f = pred_table[pred_table.student_int8_pred != pred_table.intent]
        f.to_csv(RESULTS_DIR / "failures_student_int8.csv", index=False)

    md = ["| Model | Precision | Accuracy | In-scope acc. | OOS recall | Size (MB) | Latency mean / p95 (ms) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['precision']} | {r['accuracy']:.4f} | {r['in_scope_accuracy']:.4f} | "
                  f"{r['oos_recall']:.4f} | {r['size_mb']} | {r['latency_ms_mean']} / {r['latency_ms_p95']} |")
    md.append(f"\nTest set: {len(te)} messages ({results['n_test_oos']} out-of-scope). "
              f"Latency: mean of {N_LATENCY} single-message runs incl. normalization. Hardware: {results['hardware']}")
    (RESULTS_DIR / "results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
