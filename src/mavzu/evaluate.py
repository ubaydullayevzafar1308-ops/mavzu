"""Evaluate a trained model on the test split and write a report.

    python -m mavzu.evaluate --kind baseline --path models/baseline
    python -m mavzu.evaluate --kind onnx --path models/onnx --max-samples 10000

Outputs to reports/<name>/:
  metrics.json           accuracy, macro/weighted F1, per-class scores, top confusions
  confusion_matrix.png   row-normalized confusion matrix
  robustness.json        Latin vs Cyrillic (raw) vs Cyrillic (normalized) macro-F1
  errors.csv             a sample of misclassified articles for error analysis
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mavzu.data import load_split
from mavzu.labels import LABELS, LABELS_EN
from mavzu.metrics import classification_metrics, confusion, top_confusions
from mavzu.predict import Predictor, load_predictor
from mavzu.text import latin_to_cyrillic, normalize


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [LABELS_EN[x] for x in LABELS]
    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def latency(predictor: Predictor, texts: list[str], repeats: int = 50) -> dict:
    """Single-request latency (batch of 1), the number that matters for the API."""
    sample = [texts[i % len(texts)] for i in range(repeats)]
    predictor.predict_proba(sample[:2])  # warm-up
    times = []
    for t in sample:
        t0 = time.perf_counter()
        predictor.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
    }


def robustness(predictor: Predictor, df: pd.DataFrame) -> dict:
    """How much does the model lose when the same articles arrive in Cyrillic?"""
    y = df["label"].to_numpy()
    latin = df["text"].tolist()
    cyr = [latin_to_cyrillic(t) for t in latin]
    result = {}
    for name, texts in {
        "latin": latin,
        "cyrillic_raw": cyr,  # what a naive pipeline would feed the model
        "cyrillic_normalized": [normalize(t) for t in cyr],  # our pipeline
    }.items():
        pred = predictor.predict_proba(texts).argmax(1)
        result[name] = round(classification_metrics(y, pred)["macro_f1"], 4)
    result["n"] = len(df)
    return result


def evaluate(
    kind: str, path: str, data_dir: str, max_samples: int | None, robust_n: int, out_root: Path
) -> dict:
    predictor = load_predictor(kind, path)
    name = predictor.name.replace(":", "_")
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    test = load_split(data_dir, "test", max_samples)
    texts = test["text"].tolist()  # already normalized at prepare time
    y = test["label"].to_numpy()

    t0 = time.perf_counter()
    proba = predictor.predict_proba(texts)
    elapsed = time.perf_counter() - t0
    pred = proba.argmax(1)

    metrics = classification_metrics(y, pred)
    cm = confusion(y, pred)
    metrics["top_confusions"] = top_confusions(cm, k=10)
    metrics["n_test"] = len(test)
    metrics["throughput_docs_per_s"] = round(len(test) / elapsed, 1)
    metrics["latency"] = latency(predictor, texts)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    plot_confusion(cm, out / "confusion_matrix.png", f"{name} — test macro-F1 {metrics['macro_f1']:.3f}")

    rob = robustness(predictor, load_split(data_dir, "test", robust_n, seed=7))
    (out / "robustness.json").write_text(json.dumps(rob, indent=2))

    wrong = test.assign(pred=pred, confidence=proba.max(1))[pred != y]
    wrong = wrong.sort_values("confidence", ascending=False).head(300)
    wrong.assign(
        true_label=wrong["label"].map(LABELS.__getitem__),
        pred_label=wrong["pred"].map(LABELS.__getitem__),
        text=wrong["text"].str.slice(0, 300),
    )[["true_label", "pred_label", "confidence", "text"]].to_csv(out / "errors.csv", index=False)

    print(
        f"[{name}] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"p50={metrics['latency']['p50_ms']}ms robustness={rob}"
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", required=True, choices=["baseline", "transformer", "onnx"])
    p.add_argument("--path", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--max-samples", type=int, default=None, help="evaluate on a stratified subsample")
    p.add_argument("--robust-n", type=int, default=3000)
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    evaluate(args.kind, args.path, args.data_dir, args.max_samples, args.robust_n, args.out)


if __name__ == "__main__":
    main()
