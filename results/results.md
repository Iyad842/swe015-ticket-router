| Model | Precision | Accuracy | In-scope acc. | OOS recall | Size (MB) | Latency mean / p95 (ms) |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | FP64 | 0.8659 | 0.9147 | 0.6450 | 66.9 | 37.24 / 50.88 |
| FFN on own fastText embeddings | FP32 | 0.8387 | 0.9005 | 0.5590 | 40.6 | 0.47 / 1.04 |
| mBERT fine-tuned (teacher) | FP32 | 0.8668 | 0.9368 | 0.5500 | 711.9 | 107.13 / 134.67 |
| DistilmBERT distilled (student) | FP32 | 0.8805 | 0.9332 | 0.6420 | 541.8 | 53.46 / 64.72 |
| DistilmBERT distilled + INT8 quantized | INT8 (dynamic Linear + weight-only Embedding, x86) | 0.8802 | 0.9204 | 0.6980 | 136.7 | 18.18 / 27.39 |

Test set: 5524 messages (1000 out-of-scope). Latency: mean of 200 single-message runs incl. normalization. Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 logical cores, Linux 6.6.159+, Python 3.13.16, torch 2.14.1+cu130 (1 threads), CPU only, batch size 1