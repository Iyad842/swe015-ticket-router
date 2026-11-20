"""Build the 11-slide presentation. Every number comes from results/results.json
(written by `python -m src.evaluate`), so the deck always matches YOUR run.

    python slides/make_figures.py && python slides/build_slides.py
    -> slides/SWE015_presentation.pptx   (open it, fill the [ ] fields, export to PDF)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "slides" / "figures"
RES = ROOT / "results"
OUT = ROOT / "slides" / "SWE015_presentation.pptx"

# ---- fill these in --------------------------------------------------------
STUDENT_NAME = "[Your Name Surname]"
STUDENT_ID = "[Student ID]"
REPO_URL = "[https://github.com/<you>/swe015-ticket-router]"
VIDEO_URL = "[https://youtu.be/<unlisted-video>]"
HF_URL = "[https://huggingface.co/<you>/swe015-ticket-router]"
# Success targets stated in the introduction (set BEFORE looking at results)
TARGET_ACC, TARGET_OOS, TARGET_MS = 0.85, 0.60, 50.0

NAVY, ACCENT, GREY, LIGHT = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0xC0, 0x50, 0x4D), RGBColor(0x55, 0x55, 0x55), RGBColor(0xF2, 0xF4, 0xF7)
TBD = "‹run python -m src.evaluate›"

R = json.loads((RES / "results.json").read_text()) if (RES / "results.json").exists() else None
M = {r["key"]: r for r in R["models"]} if R else {}


def pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else TBD


def num(key, field, fmt="{:.1f}"):
    return fmt.format(M[key][field]) if key in M else TBD


# --------------------------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = prs.slide_width


def text_box(slide, x, y, w, h, text="", size=16, bold=False, color=GREY, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color
    return tf


def bullets(slide, x, y, w, h, items, size=16):
    """items: list of str or (str, level)."""
    tf = slide.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        txt, lvl = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = ("• " if lvl == 0 else "– ") + txt
        r.font.size = Pt(size if lvl == 0 else size - 2)
        r.font.color.rgb = RGBColor(0x22, 0x22, 0x22) if lvl == 0 else GREY
        p.level = lvl
    return tf


def new_slide(section, title, number):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.42))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    text_box(s, Inches(0.5), Inches(0.05), Inches(6), Inches(0.35), section.upper(), 12, True, RGBColor(255, 255, 255))
    text_box(s, Inches(10.3), Inches(0.05), Inches(2.6), Inches(0.35), "SWE015 · Ticket Router", 11,
             False, RGBColor(0xDD, 0xE6, 0xF0), PP_ALIGN.RIGHT)
    text_box(s, Inches(0.5), Inches(0.6), Inches(12.3), Inches(0.8), title, 28, True, NAVY)
    text_box(s, Inches(12.4), Inches(7.05), Inches(0.7), Inches(0.35), str(number), 11, False, GREY, PP_ALIGN.RIGHT)
    return s


def table(slide, x, y, w, data, col_w=None, size=12, row_h=0.36, header_color=NAVY, highlight_row=None):
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, x, y, w, Inches(row_h * rows))
    t = shp.table
    if col_w:
        tot = sum(col_w)
        for j, cw in enumerate(col_w):
            t.columns[j].width = Emu(int(w * cw / tot))
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = str(val)
            p = c.text_frame.paragraphs[0]
            p.font.size = Pt(size)
            p.font.bold = i == 0 or (highlight_row is not None and i == highlight_row)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            if i == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = header_color
                p.font.color.rgb = RGBColor(255, 255, 255)
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = RGBColor(0xFD, 0xEC, 0xEA) if i == highlight_row else (
                    LIGHT if i % 2 else RGBColor(255, 255, 255))
    return t


def note(slide, text, y=6.75):
    text_box(slide, Inches(0.5), Inches(y), Inches(11.8), Inches(0.5), text, 10.5, False, GREY)


# ---- 1. title ---------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(4.2))
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
text_box(s, Inches(0.8), Inches(1.0), Inches(11.5), Inches(1.2), "Customer Support Ticket Router", 44, True, RGBColor(255, 255, 255))
text_box(s, Inches(0.8), Inches(2.1), Inches(11.5), Inches(1.4),
         "Routing Turkish & English support messages with a classical baseline, a neural network,\n"
         "a fine-tuned multilingual BERT and a distilled, INT8-quantized Transformer", 18, False, RGBColor(0xDD, 0xE6, 0xF0))
text_box(s, Inches(0.8), Inches(3.4), Inches(11.5), Inches(0.5),
         "SWE015 – Introduction to Large Language Models · Homework · Use Case 2", 14, False, RGBColor(0xB8, 0xC8, 0xDC))
info = [("Student", STUDENT_NAME), ("Student ID", STUDENT_ID), ("Use case", "2 – Customer Support Ticket Router"),
        ("Repository", REPO_URL), ("Demo video", VIDEO_URL), ("Model weights", HF_URL)]
table(s, Inches(0.8), Inches(4.6), Inches(11.7), [["Item", "Details"]] + [list(r) for r in info],
      col_w=[1.4, 6], size=14, row_h=0.36)

# ---- 2. introduction -----------------------------------------------------------
s = new_slide("Introduction", "Problem, users and what success looks like", 2)
bullets(s, Inches(0.5), Inches(1.5), Inches(6.3), Inches(5), [
    "Problem: a home-appliance & services brand receives thousands of free-text support "
    "messages per day in Turkish and English, e.g. “bulaşık makinem su boşaltmıyor, model DW-6534”.",
    "Manual triage is slow; every message must reach the right team instantly.",
    "Users: customers (fast answer) and support agents (only get tickets meant for them).",
    "Use case 2: free-text message → normalize → intent + confidence → routed team.",
    ("Low-confidence / out-of-scope messages → human agent", 1),
    ("Greetings & small talk → rule-based (ELIZA) reply", 1),
    ("All four model types shown side by side in the app", 1),
], 16)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.6), Inches(4.4))
box.fill.solid()
box.fill.fore_color.rgb = LIGHT
box.line.color.rgb = NAVY
text_box(s, Inches(7.45), Inches(1.75), Inches(5.2), Inches(0.5), "Success criteria (set before training)", 18, True, NAVY)
bullets(s, Inches(7.45), Inches(2.35), Inches(5.2), Inches(3.6), [
    f"Test accuracy ≥ {TARGET_ACC:.0%} (157 classes incl. out-of-scope)",
    f"Out-of-scope recall ≥ {TARGET_OOS:.0%} – unknown requests must reach a human",
    f"< {TARGET_MS:.0f} ms per message on a laptop CPU with the quantized model",
    "Quantized model clearly smaller and faster than the full Transformer with only a small accuracy loss",
    "Works for new, typed input in a real application",
], 15)

# ---- 3. data -------------------------------------------------------------------
s = new_slide("Development", "Data: sources, size, splits, preprocessing", 3)
counts = {}
for sp in ("train", "val", "test"):
    p = ROOT / "data" / "processed" / f"{sp}.csv"
    if p.exists():
        d = pd.read_csv(p, keep_default_na=False)
        counts[sp] = (len(d), int((d.intent == "oos").sum()), int((d.domain == "appliance").sum()), int((d.lang == "tr").sum()))
rows = [["Split", "Messages", "Out-of-scope", "Appliance (hand-written)", "Turkish"]]
for sp, name in (("train", "Train"), ("val", "Validation"), ("test", "Test (held out)")):
    rows.append([name, *counts[sp]] if sp in counts else [name, "‹run src.data›", "", "", ""])
table(s, Inches(0.5), Inches(1.5), Inches(7.2), rows, col_w=[1.6, 1.2, 1.3, 2, 1], size=13)
bullets(s, Inches(0.5), Inches(3.2), Inches(7.2), Inches(3.4), [
    "CLINC150 “oos_plus” (Larson et al., EMNLP 2019): 150 intents in 10 domains + an out-of-scope class (English).",
    "+ 96 hand-written appliance messages (6 intents, TR + EN, with model codes and typos), split 10 / 2 / 4 per intent.",
    "157 classes in total; the test split is never used for training or threshold tuning.",
], 14)
text_box(s, Inches(8.1), Inches(1.5), Inches(4.8), Inches(0.5), "Preprocessing pipeline", 18, True, NAVY)
bullets(s, Inches(8.1), Inches(2.05), Inches(4.8), Inches(4.6), [
    "Regex masking of URLs / e-mails, squeezing repeated letters (“pleaseee”)",
    "Regex model-code extraction DW-6534 → MODELCODE, error codes (E21) shown in the app",
    "Minimum-edit-distance correction against a TR/EN product lexicon: dishwaher → dishwasher, bulasik → bulaşık (only for unknown words)",
    "Turkish-aware lower-casing + tokenization for TF-IDF / embeddings",
    "Cased text + WordPiece tokenizer for the Transformers",
], 14)

# ---- 4. architecture ---------------------------------------------------------
s = new_slide("Development", "Architecture: from user input to routed team", 4)
if (FIG / "architecture.png").exists():
    pic = s.shapes.add_picture(str(FIG / "architecture.png"), 0, Inches(1.35), height=Inches(5.5))
    pic.left = int((W - pic.width) / 2)
note(s, "Every box is a file in the repository. The app (python app.py) runs fully on CPU; training ran offline on a Colab GPU.", 7.0)

# ---- 5. technique map ----------------------------------------------------------
s = new_slide("Development", "Technique map (R2)", 5)
tm = [["Area", "Technique", "Component in the application", "Code"],
      ["A · Neural networks", "Feed-forward network, SGD + backpropagation, dropout", "FFN intent classifier on averaged embeddings", "src/ffn.py"],
      ["A / B", "fastText-style embeddings (skip-gram, negative sampling, char n-grams)", "Own word vectors trained on the training messages", "src/embeddings.py"],
      ["B · Classical NLP", "Regular expressions, normalization, tokenization", "Message normalization, model-code & error-code extraction", "src/text_utils.py"],
      ["B · Classical NLP", "Minimum edit distance", "Spelling correction of product names", "src/text_utils.py"],
      ["B · Classical NLP", "ELIZA-style rules + pronoun reflection", "Greeting / small-talk replies", "src/eliza.py"],
      ["B · Classical NLP", "TF-IDF + logistic regression", "Classical baseline", "src/baseline.py"],
      ["C · Transformers", "Fine-tuning multilingual BERT with a classification head; WordPiece", "Teacher intent classifier", "src/transformer.py"],
      ["D · Efficient models", "Knowledge distillation (teacher → 6-layer student)", "Student intent classifier", "src/transformer.py"],
      ["D · Efficient models", "Post-training dynamic INT8 quantization; size/latency benchmark", "Primary model used by the app", "src/transformer.py, src/evaluate.py"]]
table(s, Inches(0.5), Inches(1.45), Inches(12.3), tm, col_w=[1.6, 3.6, 3.4, 2.2], size=12, row_h=0.47)

# ---- 6. models ---------------------------------------------------------------
s = new_slide("Development", "The model ladder and how each model was trained", 6)
md = [["#", "Model", "Input", "Architecture", "Training"],
      ["1", "TF-IDF + LogReg", "lower-cased tokens", "word 1–2 + char 3–5-gram TF-IDF → multinomial logistic regression (C = 10)", "scikit-learn, CPU, seconds"],
      ["2", "fastText + FFN", "averaged 100-d word vectors", "own skip-gram embeddings (char 3–5-grams, 100k buckets) → Linear 512 → ReLU → Dropout 0.3 → Linear 157",
       "embeddings: 5 epochs, SparseAdam; FFN: SGD lr 0.1, momentum 0.9, cosine decay, 40 epochs, best val epoch"],
      ["3", "mBERT (teacher)", "cased text, WordPiece, ≤ 64 tokens", "bert-base-multilingual-cased, 12 layers + linear head",
       "AdamW lr 4e-5, 4 epochs, batch 32, warm-up 10 %, T4 GPU"],
      ["4", "DistilmBERT (student)", "same tokenizer", "distilbert-base-multilingual-cased, 6 layers + head",
       "knowledge distillation: 0.5·T²·KL(teacher‖student, T = 2) + 0.5·CE, AdamW lr 5e-5, 5 epochs"],
      ["5", "Student INT8", "same", "student with int8 Linear weights (dynamic quantization)", "no re-training (post-training quantization)"]]
table(s, Inches(0.5), Inches(1.45), Inches(12.3), md, col_w=[0.3, 1.7, 1.8, 4.3, 4.2], size=12, row_h=0.62)
note(s, "The out-of-scope decision of every model: predict ‘oos’ if the top class is ‘oos’ or its probability < τ; τ is chosen on the validation set (grid 0–0.95).", 6.0)

# ---- 7. quantization -------------------------------------------------------
s = new_slide("Development", "Efficient model: distillation + INT8 quantization", 7)
bullets(s, Inches(0.5), Inches(1.5), Inches(6.4), Inches(5), [
    "Step 1 – Knowledge distillation: the 12-layer mBERT teacher’s soft probabilities (T = 2) train a 6-layer DistilmBERT student that uses the same WordPiece vocabulary.",
    "Step 2 – Post-training dynamic quantization (PyTorch torch.ao.quantization.quantize_dynamic):",
    ("Linear-layer weights stored as int8, activations quantized on the fly at run time", 1),
    ("Embeddings: int8 weight-only if supported, otherwise FP32 (exact setting in the table)", 1),
    ("No calibration data and no re-training needed; runs on CPU (x86 fbgemm / ARM qnnpack)", 1),
    f"Precision reported by the code: {M['student_int8']['precision'] if 'student_int8' in M else TBD}",
], 15)
q = [["Model", "Size (MB)", "Latency (ms)", "Accuracy"]]
for k in ("teacher", "student", "student_int8"):
    q.append([M[k]["model"] if k in M else k, num(k, "size_mb"), num(k, "latency_ms_mean", "{:.2f}"),
              pct(M[k]["accuracy"]) if k in M else TBD])
table(s, Inches(7.2), Inches(1.6), Inches(5.6), q, col_w=[2.6, 1, 1.1, 1], size=12, row_h=0.5, highlight_row=3)
if M.get("teacher") and M.get("student_int8"):
    t, i8 = M["teacher"], M["student_int8"]
    text_box(s, Inches(7.2), Inches(3.9), Inches(5.6), Inches(2),
             f"INT8 student vs. teacher: {t['size_mb'] / i8['size_mb']:.1f}× smaller, "
             f"{t['latency_ms_mean'] / i8['latency_ms_mean']:.1f}× faster, "
             f"accuracy change {100 * (i8['accuracy'] - t['accuracy']):+.1f} points.", 15, True, ACCENT)

# ---- 8. results --------------------------------------------------------------
s = new_slide("Development", "Results on the same held-out test set (R4)", 8)
hdr = ["Model", "Precision", "Accuracy", "In-scope acc.", "OOS recall", "τ", "Size (MB)", "Latency (ms)"]
rows = [hdr]
order = ["baseline", "ffn", "teacher", "student", "student_int8"]
for k in order:
    if k in M:
        r = M[k]
        rows.append([r["model"], r["precision"].split(" (")[0], pct(r["accuracy"]), pct(r["in_scope_accuracy"]),
                     pct(r["oos_recall"]), f"{r['threshold']:.2f}", f"{r['size_mb']:.1f}", f"{r['latency_ms_mean']:.2f}"])
    else:
        rows.append([k, "", TBD, "", "", "", "", ""])
table(s, Inches(0.5), Inches(1.45), Inches(12.3), rows, col_w=[3.3, 0.9, 1, 1.1, 1, 0.6, 1, 1.1], size=13,
      row_h=0.45, highlight_row=5)
if (FIG / "results_chart.png").exists():
    s.shapes.add_picture(str(FIG / "results_chart.png"), Inches(1.6), Inches(4.3), height=Inches(2.35))
hw = (f"Test set: {R['n_test']} messages ({R['n_test_oos']} out-of-scope, {R['n_test_appliance']} appliance). "
      f"Latency: mean of {R['n_latency_runs']} single-message runs incl. normalization, batch 1. Hardware: {R['hardware']}.") if R else \
    "Hardware / latency details are filled in automatically after running src.evaluate."
note(s, hw, 6.7)

# ---- 9. conclusion: comparison ---------------------------------------------------
s = new_slide("Conclusion", "What the comparison showed", 9)
if M:
    best = max(M.values(), key=lambda r: r["accuracy"])
    fastest = min(M.values(), key=lambda r: r["latency_ms_mean"])
    best_oos = max(M.values(), key=lambda r: r["oos_recall"])
    items = [f"Highest accuracy: {best['model']} ({pct(best['accuracy'])}); fastest: {fastest['model']} "
             f"({fastest['latency_ms_mean']:.2f} ms); best out-of-scope recall: {best_oos['model']} ({pct(best_oos['oos_recall'])})."]
    if "baseline" in M and "teacher" in M:
        b, t = M["baseline"], M["teacher"]
        d = 100 * (t["accuracy"] - b["accuracy"])
        items.append(f"Transformer vs. TF-IDF baseline: {d:+.1f} accuracy points. " + (
            "Contextual multilingual representations beat sparse n-gram features." if d > 0 else
            "The strong char-n-gram baseline is hard to beat on short, keyword-heavy CLINC queries."))
        items.append(f"…but the baseline is {t['latency_ms_mean'] / b['latency_ms_mean']:.0f}× faster and "
                     f"{t['size_mb'] / max(b['size_mb'], 0.1):.0f}× smaller than the teacher.")
    if "ffn" in M and "baseline" in M:
        f = M["ffn"]
        items.append(f"FFN on averaged embeddings: {pct(f['accuracy'])} – averaging loses word order and the embeddings were trained on only "
                     f"~15k short sentences, so it " + ("trails" if f["accuracy"] < M["baseline"]["accuracy"] else "beats") + " the TF-IDF baseline.")
    if "student" in M and "student_int8" in M:
        st, i8 = M["student"], M["student_int8"]
        items.append(f"INT8 quantization of the student: accuracy {100 * (i8['accuracy'] - st['accuracy']):+.1f} points, "
                     f"latency {st['latency_ms_mean']:.1f} → {i8['latency_ms_mean']:.1f} ms, size {st['size_mb']:.0f} → {i8['size_mb']:.0f} MB.")
    if "student_int8" in M:
        i8 = M["student_int8"]
        ok = [i8["accuracy"] >= TARGET_ACC, i8["oos_recall"] >= TARGET_OOS, i8["latency_ms_mean"] < TARGET_MS]
        items.append("Success criteria for the deployed INT8 model: accuracy " + ("met" if ok[0] else "NOT met") +
                     ", OOS recall " + ("met" if ok[1] else "NOT met") + ", latency " + ("met" if ok[2] else "NOT met") + ".")
    sub = R.get("subsets", {})
    if "student_int8" in sub and "baseline" in sub:
        items.append(f"Hand-written appliance test messages: INT8 {pct(sub['student_int8']['appliance_accuracy'])} vs. baseline "
                     f"{pct(sub['baseline']['appliance_accuracy'])}; Turkish messages: INT8 {pct(sub['student_int8']['turkish_accuracy'])} "
                     f"vs. baseline {pct(sub['baseline']['turkish_accuracy'])} (small sample – interpret with care).")
else:
    items = [TBD + " – this slide is generated from your results."]
bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(5.3), items, 16)
note(s, "Check these auto-generated sentences against your numbers and rewrite them in your own words before submitting.", 6.85)

# ---- 10. limitations / failure / next --------------------------------------------
s = new_slide("Conclusion", "Limitations, a failure example and next steps", 10)
fail_txt = TBD
fp = RES / "failures_student_int8.csv"
if fp.exists():
    f = pd.read_csv(fp, keep_default_na=False)
    if len(f):
        pick = f[(f.domain == "appliance") | (f.lang == "tr")]
        pick = pick if len(pick) else f[f.intent == "oos"] if (f.intent == "oos").any() else f
        r = pick.iloc[0]
        fail_txt = (f"Input: “{r['text']}”\nExpected: {r['intent']}   →   INT8 predicted: {r['student_int8_pred']} "
                    f"(confidence {float(r['student_int8_conf']):.2f})")
bullets(s, Inches(0.5), Inches(1.5), Inches(6.2), Inches(5), [
    "CLINC150 is English and not about appliances – the Turkish / appliance part rests on only 96 hand-written messages (24 in test).",
    "Out-of-scope detection relies on one global confidence threshold; unseen topics phrased like known intents still get routed.",
    "Dynamic quantization only shrinks Linear layers; the large multilingual embedding matrix dominates model size.",
    "Latency measured on one machine with batch size 1; numbers change with hardware and thread count.",
], 15)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.55), Inches(5.8), Inches(2.2))
box.fill.solid()
box.fill.fore_color.rgb = RGBColor(0xFD, 0xEC, 0xEA)
box.line.color.rgb = ACCENT
text_box(s, Inches(7.2), Inches(1.65), Inches(5.4), Inches(0.4), "Failure example (from the test set)", 16, True, ACCENT)
text_box(s, Inches(7.2), Inches(2.1), Inches(5.4), Inches(1.6), fail_txt, 13, False, RGBColor(0x33, 0x33, 0x33))
text_box(s, Inches(7.0), Inches(3.95), Inches(5.8), Inches(0.4), "Next steps", 16, True, NAVY)
bullets(s, Inches(7.0), Inches(4.4), Inches(5.8), Inches(2.4), [
    "Train on MASSIVE tr-TR + more real Turkish appliance tickets",
    "Vocabulary pruning / ONNX Runtime INT8 to shrink the embedding matrix",
    "Better OOS detection (energy score, Mahalanobis distance)",
    "Use the extracted model code to look up the product and warranty",
], 14)

# ---- 11. AI usage ------------------------------------------------------------
s = new_slide("Conclusion", "AI usage statement (R8)", 11)
bullets(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(4.8), [
    "Tool used: Claude (Anthropic) as an AI coding and writing assistant.   [add any other tools you used]",
    "Used for: initial code scaffolding of the training / evaluation scripts and the Gradio app, the slide-generation script, "
    "drafting the architecture diagram and slide text, and drafting part of the 96 appliance example messages.",
    "Done by me: choosing the use case and design, reviewing / editing / debugging every file, running all training and "
    "evaluation, checking the hand-written data, and writing the conclusions.   [edit to match what you actually did]",
    "All numbers in this presentation were produced by my own runs of src/evaluate.py (results/results.json in the repository); "
    "none were taken from an AI answer, paper or model card.",
    "I can explain every component in the architecture diagram.",
], 17)

prs.save(OUT)
print("wrote", OUT, "| results found:" if R else "| NO results yet (placeholders)", bool(R))
