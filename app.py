"""Gradio demo — runs locally and as a Hugging Face Space.

MODEL_KIND=onnx MODEL_PATH=models/onnx python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402

from mavzu.labels import LABELS_EN  # noqa: E402
from mavzu.predict import classify, load_predictor  # noqa: E402
from mavzu.text import normalize  # noqa: E402

predictor = load_predictor(os.getenv("MODEL_KIND", "onnx"), os.getenv("MODEL_PATH", "models/onnx"))

EXAMPLES = [
    ["O‘zbekiston terma jamoasi jahon chempionati saralash bosqichida g‘alaba qozondi."],
    ["Yangi smartfon kuchli kamera va tez quvvatlanish funksiyasi bilan taqdim etildi."],
    ["Markaziy bank asosiy stavkani o‘zgarishsiz qoldirdi, inflyatsiya sekinlashmoqda."],
    ["Шифокорлар кузда грипп касаллигидан эмланишни тавсия қилмоқда."],
    ["Palov tayyorlash uchun guruch, sabzi, go‘sht va zirvak kerak bo‘ladi."],
    ["Тошкентда янги автомобиль заводи қурилиши бошланди."],
]


def run(text: str):
    if not text or not text.strip():
        return {}, "", ""
    pred = classify(predictor, [text], top_k=5)[0]
    scores = {f"{s['label']} ({s['label_en']})": s["score"] for s in pred.top_k}
    return scores, pred.script, normalize(text)


with gr.Blocks(title="Mavzu — Uzbek news topic classifier") as demo:
    gr.Markdown(
        "# Mavzu 🇺🇿\n"
        "Classifies Uzbek text into 15 news topics. Works with **Latin and Cyrillic** script — "
        "Cyrillic input is transliterated before it reaches the model.\n\n"
        f"Topics: {', '.join(LABELS_EN.values())}"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(lines=6, label="Matn / Text", placeholder="Yangilik matnini kiriting…")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out = gr.Label(num_top_classes=5, label="Topic")
            script = gr.Textbox(label="Detected script")
            norm = gr.Textbox(label="Normalized input (what the model sees)", lines=3)
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(run, inputs=inp, outputs=[out, script, norm])
    inp.submit(run, inputs=inp, outputs=[out, script, norm])

if __name__ == "__main__":
    demo.launch()
