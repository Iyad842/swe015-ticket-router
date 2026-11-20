# SWE015 – Customer Support Ticket Router (Use Case 2)

An intelligent application that routes free-text customer support messages
(Turkish and English) to the right team. It predicts the intent with a
confidence score, sends low-confidence / out-of-scope messages to a
**human agent**, and answers greetings and small talk with an **ELIZA-style rule
bot**. The application runs an **INT8-quantized, distilled multilingual BERT**
on CPU and can show all five models of the ladder side by side.

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

## 1. Run the application (single command)

```bash
git clone https://github.com/Iyad842/swe015-ticket-router.git
cd swe015-ticket-router
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ROUTER_HF_REPO=IYAD842/swe015-ticket-router   # Windows: set ROUTER_HF_REPO=...
python app.py
```

Open http://127.0.0.1:7860. On the first start the trained weights are
downloaded from the Hugging Face Hub (`https://huggingface.co/<YOUR_HF_USERNAME>/swe015-ticket-router`).
Instead of the environment variable you can also edit `HF_REPO_ID` in `src/config.py`.

## 2. Reproduce training and evaluation

Training the two Transformers needs a GPU (a free Google Colab T4 is enough,
roughly 15–25 minutes). See `notebooks/train_on_colab.ipynb`, or:

```bash
python -m src.train_all          # data -> baseline -> embeddings -> FFN -> teacher -> student -> evaluation
python -m src.hub push           # upload ./models to the Hugging Face Hub (after `huggingface-cli login`)
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

## 3. Build the presentation from your own results

```bash
python slides/make_figures.py    # architecture diagram + results chart -> slides/figures/
python slides/build_slides.py    # -> slides/SWE015_presentation.pptx (numbers read from results/results.json)
```

Open the `.pptx`, fill in the name/ID/link placeholders, check the text, and export as PDF.

## 4. Repository layout

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
slides/                 figure + slide generators
tests/                  unit tests (python -m pytest -q)
```

## 5. Data and models (sources)

* **CLINC150** (`data_oos_plus`): S. Larson et al., *An Evaluation Dataset for
  Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP 2019.
  https://github.com/clinc/oos-eval — 150 intents in 10 domains + out-of-scope class.
* **Appliance domain**: 96 hand-written messages (6 intents, Turkish + English),
  `data/appliance_extra.json`, split 10/2/4 per intent (train/val/test).
* `bert-base-multilingual-cased` (Devlin et al., 2019) and
  `distilbert-base-multilingual-cased` (Sanh et al., 2019) from the Hugging Face Hub.
* Knowledge distillation: Hinton, Vinyals & Dean, 2015.
