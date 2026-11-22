# SWE015 – Customer Support Ticket Router (Use Case 2)

An intelligent application that routes free-text customer support messages
(Turkish and English) to the right team. It predicts the intent with a
confidence score, sends low-confidence / out-of-scope messages to a
**human agent**, and answers greetings and small talk with an **ELIZA-style rule
bot**. The application runs an **INT8-quantized, distilled multilingual BERT**
on CPU and can show all five models of the ladder side by side.

* Code: https://github.com/Iyad842/swe015-ticket-router
* Model weights: https://huggingface.co/IYAD842/swe015-ticket-router
* Demo video: https://youtu.be/_Byy3nLA4wU

| Ladder step | Model | Technique area |
|---|---|---|
| Classical baseline | TF-IDF (word 1–2 + char 3–5 grams) + logistic regression | B |
| Neural network | Own fastText-style embeddings (skip-gram, neg. sampling) → averaged → feed-forward net, SGD + backprop | A (+B) |
| Transformer | `bert-base-multilingual-cased` fine-tuned with a classification head (teacher) | C |
| Efficient | `distilbert-base-multilingual-cased` trained by **knowledge distillation**, then **dynamic INT8 quantization** | D |

Preprocessing (area B): regex normalization (URLs, e-mails, repeated letters),
regex **model-code extraction** (`DW-6534`) and error-code extraction (`E21`),
**minimum-edit-distance** correction of misspelled product names
(`dishwaher → dishwasher`, `bulasik → bulaşık`), ELIZA rules for small talk.

---

## 1. Run the application

**Requirement:** Python 3.10 or newer from https://www.python.org/downloads/
(on Windows, tick **"Add python.exe to PATH"** in the installer).
No GPU is needed to run the app.

### Windows (PowerShell)

```powershell
git clone https://github.com/Iyad842/swe015-ticket-router.git
cd swe015-ticket-router
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:ROUTER_HF_REPO="IYAD842/swe015-ticket-router"
python app.py
```

No git installed? Download the ZIP instead:

```powershell
Invoke-WebRequest https://github.com/Iyad842/swe015-ticket-router/archive/refs/heads/main.zip -OutFile repo.zip
Expand-Archive repo.zip -DestinationPath .
cd swe015-ticket-router-main
```

then continue from the `Set-ExecutionPolicy` line above.

### macOS / Linux

```bash
git clone https://github.com/Iyad842/swe015-ticket-router.git
cd swe015-ticket-router
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ROUTER_HF_REPO=IYAD842/swe015-ticket-router
python app.py
```

Open **http://127.0.0.1:7860**. On the first start the trained weights (about 1.3 GB)
are downloaded automatically from the Hugging Face Hub
(https://huggingface.co/IYAD842/swe015-ticket-router) into `./models`.
Instead of the environment variable you can also set `HF_REPO_ID` in `src/config.py`.

**Troubleshooting:** if the app says a model file is missing, delete the `models`
folder and start `python app.py` again (it downloads the weights again).
Avoid running the project inside a OneDrive-synced folder.

## 2. Reproduce training and evaluation

Training the two Transformers needs a GPU (a free Google Colab T4 is enough,
roughly 15–25 minutes). See `notebooks/train_on_colab.ipynb`, or:

```bash
python -m src.train_all          # data -> baseline -> embeddings -> FFN -> teacher -> student -> evaluation
python -m src.hub push           # upload ./models to the Hugging Face Hub (after `hf auth login`)
```

Single steps: `python -m src.data`, `python -m src.baseline`, `python -m src.embeddings`,
`python -m src.ffn`, `python -m src.transformer teacher`, `python -m src.transformer student`,
`python -m src.evaluate`.

`python -m src.evaluate` evaluates **all five models on the same held-out test
set** and writes `results/results.json`, `results/results.md`,
`results/test_predictions.csv` and `results/failures_student_int8.csv`.
The out-of-scope confidence threshold of each model is tuned on the
validation set only. Latency = mean of 200 single-message CPU runs including
normalization (after 20 warm-up runs).

## 3. Results (from my own run, see `results/results.json`)

Test set: 5524 messages (1000 out-of-scope, 24 hand-written appliance messages).
Latency measured on an Intel Xeon CPU @ 2.00 GHz (Google Colab, CPU only, batch size 1).

| Model | Precision | Accuracy | In-scope acc. | OOS recall | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | FP64 | 86.6% | 91.5% | 64.5% | 66.9 | 37.24 |
| FFN on own fastText embeddings | FP32 | 83.9% | 90.1% | 55.9% | 40.6 | 0.47 |
| mBERT fine-tuned (teacher) | FP32 | 86.7% | 93.7% | 55.0% | 711.9 | 107.13 |
| DistilmBERT distilled (student) | FP32 | 88.1% | 93.3% | 64.2% | 541.8 | 53.46 |
| **DistilmBERT distilled + INT8 quantized** | **INT8** | **88.0%** | **92.0%** | **69.8%** | **136.7** | **18.18** |

## 4. Build the presentation from the results

```bash
python slides/make_figures.py    # architecture diagram + results chart -> slides/figures/
python slides/build_slides.py    # -> slides/SWE015_presentation.pptx (numbers read from results/results.json)
```

## 5. Repository layout

```
app.py                  Gradio application (uses the INT8 model)
src/config.py           paths, model names, hyper-parameters
src/text_utils.py       regex normalization, model codes, edit-distance correction
src/eliza.py            ELIZA-style small-talk rules
src/routing.py          intent -> domain -> team
src/data.py             CLINC150 download + hand-written appliance data + splits
src/baseline.py         TF-IDF + logistic regression
src/embeddings.py       own fastText-style skip-gram embeddings
src/ffn.py              feed-forward network (SGD + backprop)
src/transformer.py      mBERT teacher, distillation, INT8 quantization
src/predictors.py       common predict interface for all models
src/evaluate.py         results table: accuracy, OOS recall, size, latency
src/hub.py              push / pull weights to the Hugging Face Hub
data/appliance_extra.json  96 hand-written TR/EN appliance messages
results/                evaluation outputs of my run
slides/                 figure + slide generators
notebooks/              Colab training notebook
tests/                  unit tests (python -m pytest -q)
```

## 6. Data and models (sources)

* **CLINC150** (`data_oos_plus`): S. Larson et al., *An Evaluation Dataset for
  Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019.
  https://github.com/clinc/oos-eval — 150 intents in 10 domains + out-of-scope class.
* **Appliance domain**: 96 hand-written messages (6 intents, Turkish + English),
  `data/appliance_extra.json`, split 10/2/4 per intent (train/val/test).
* `bert-base-multilingual-cased` (Devlin et al., 2019) and
  `distilbert-base-multilingual-cased` (Sanh et al., 2019) from the Hugging Face Hub.
* Knowledge distillation: Hinton, Vinyals & Dean, 2015.

## 7. AI usage

Claude (Anthropic) was used as a coding and writing assistant (code scaffolding,
slide generation, debugging setup problems). All training, evaluation and the
numbers above come from my own runs.
