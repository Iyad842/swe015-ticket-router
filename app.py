"""Customer Support Ticket Router - Gradio application.

    python app.py        ->  open http://127.0.0.1:7860

Pipeline: message -> regex normalization + model-code extraction + edit-distance
spelling correction -> ELIZA small-talk check -> INT8 quantized DistilmBERT
(primary model) -> confidence threshold -> team routing.
The other four models of the ladder can be shown side by side.
"""
from __future__ import annotations

import time

import gradio as gr
import pandas as pd

from src.hub import ensure_models

ensure_models()

from src.config import MODELS_DIR, OOS_LABEL  # noqa: E402
from src.eliza import eliza_reply, small_talk_reply  # noqa: E402
from src.predictors import (ALL_KEYS, PRIMARY_KEY, decide, load_labels,  # noqa: E402
                            load_predictors, load_thresholds, set_threads)
from src.routing import route  # noqa: E402
from src.text_utils import Normalizer  # noqa: E402

set_threads()
LABELS = load_labels()
NORMALIZER = Normalizer.from_file(MODELS_DIR / "vocab.json")
PREDICTORS = load_predictors(ALL_KEYS, device="cpu")
THRESHOLDS = load_thresholds()
if PRIMARY_KEY not in PREDICTORS:
    raise SystemExit("The quantized student model could not be loaded - see the messages above.")

COLUMNS = ["Model", "Precision", "Intent", "Confidence", "Routed team", "Latency (ms)"]


def analyze(message: str, compare: bool):
    if not message or not message.strip():
        return "Please type a message.", {}, "", pd.DataFrame(columns=COLUMNS)

    t_norm = time.perf_counter()
    n = NORMALIZER(message)
    norm_ms = (time.perf_counter() - t_norm) * 1000
    smalltalk = small_talk_reply(message)

    rows, primary = [], None
    for key in (ALL_KEYS if compare else [PRIMARY_KEY]):
        p = PREDICTORS.get(key)
        if p is None:
            continue
        t0 = time.perf_counter()
        probs = p.predict_proba([n])[0]
        ms = (time.perf_counter() - t0) * 1000 + norm_ms
        tau = THRESHOLDS.get(key, 0.5)
        final, conf, top = decide(probs, LABELS, tau)
        domain, team = route(final)
        rows.append([p.name, p.precision, final, round(conf, 3), team, round(ms, 1)])
        if key == PRIMARY_KEY:
            primary = dict(final=final, conf=conf, top=top, tau=tau, domain=domain, team=team, probs=probs)

    tr = n["lang"] == "tr"
    if smalltalk:
        team, reply = "Chatbot (rule-based reply)", smalltalk
        why = "ELIZA rule matched a greeting / small-talk message"
    elif primary["final"] == OOS_LABEL:
        team = "Human agent"
        reply = ("Sizi bir müşteri temsilcisine bağlıyorum." if tr
                 else "I'm connecting you to a human agent.")
        why = (f"predicted out-of-scope" if primary["top"] == OOS_LABEL
               else f"confidence {primary['conf']:.2f} < threshold {primary['tau']:.2f}")
    elif primary["domain"] == "small_talk":
        team, reply = primary["team"], eliza_reply(message, n["lang"])
        why = "intent belongs to the small-talk domain -> ELIZA reply"
    else:
        team = primary["team"]
        reply = (f"Talebiniz **{team}** ekibine yönlendirildi." if tr
                 else f"Your request was routed to **{team}**.")
        why = f"confidence {primary['conf']:.2f} >= threshold {primary['tau']:.2f}"

    decision = (f"### Routed to: {team}\n"
                f"**Intent (INT8 DistilmBERT):** `{primary['final']}` · confidence **{primary['conf']:.2f}** "
                f"· domain `{primary['domain']}`  \n"
                f"**Why:** {why}  \n**Bot reply:** {reply}")

    top = primary["probs"].argsort()[::-1][:5]
    label_dict = {LABELS[i]: float(primary["probs"][i]) for i in top}

    corr = ", ".join(f"`{a}` → `{b}`" for a, b in n["corrections"]) or "–"
    prep = (f"**Language:** {n['lang']}  \n**Normalized:** {n['clean']}  \n"
            f"**Model code(s):** {', '.join(n['model_codes']) or '–'}  \n"
            f"**Error code(s):** {', '.join(n['error_codes']) or '–'}  \n"
            f"**Spelling corrections (edit distance):** {corr}")
    return decision, label_dict, prep, pd.DataFrame(rows, columns=COLUMNS)


EXAMPLES = [
    ["bulaşık makinem su boşaltmıyor, model DW-6534", True],
    ["My dishwaher shows error E21 and stopped", True],
    ["I want to book a technician for my washing machine next week", True],
    ["how do I transfer money to my savings account", True],
    ["what's the best way to learn to play the violin", True],
    ["merhaba", True],
    ["I feel really angry, nothing works", True],
]

with gr.Blocks(title="SWE015 · Support Ticket Router") as demo:
    gr.Markdown("# Customer Support Ticket Router\nType a support message in **Turkish or English**. "
                "The primary model is an **INT8-quantized distilled multilingual BERT** running on CPU.")
    with gr.Row():
        with gr.Column(scale=3):
            msg = gr.Textbox(label="Customer message", lines=3,
                             placeholder="e.g. çamaşır makinem su boşaltmıyor, model WM-8020")
            with gr.Row():
                compare = gr.Checkbox(value=True, label="Compare all 5 models side by side")
                btn = gr.Button("Route message", variant="primary")
            decision = gr.Markdown()
        with gr.Column(scale=2):
            top5 = gr.Label(label="Top-5 intents (INT8 model)", num_top_classes=5)
    prep = gr.Markdown()
    table = gr.Dataframe(headers=COLUMNS, label="Model ladder comparison", interactive=False)
    gr.Examples(EXAMPLES, inputs=[msg, compare])
    btn.click(analyze, [msg, compare], [decision, top5, prep, table])
    msg.submit(analyze, [msg, compare], [decision, top5, prep, table])

if __name__ == "__main__":
    demo.launch()
