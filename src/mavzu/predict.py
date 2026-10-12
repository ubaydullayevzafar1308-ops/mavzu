"""Inference: one interface over the baseline, the PyTorch model and the ONNX export.

python -m mavzu.predict --kind onnx --path models/onnx "Ўзбекистон терма жамоаси ғалаба қозонди"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from mavzu.labels import LABELS, LABELS_EN
from mavzu.text import detect_script, normalize


class Predictor(Protocol):
    name: str

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), len(LABELS)); texts are already normalized."""
        ...


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class BaselinePredictor:
    name = "baseline"

    def __init__(self, path: str | Path):
        import joblib

        path = Path(path)
        self.pipe = joblib.load(path / "model.joblib" if path.is_dir() else path)
        self.classes = np.asarray(self.pipe.classes_)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raw = self.pipe.predict_proba(texts)
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        out[:, self.classes] = raw  # classes_ may be a subset in tiny test fixtures
        return out


class TransformerPredictor:
    name = "transformer"

    def __init__(self, path: str | Path, max_length: int = 256, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from mavzu.train_transformer import pick_device

        self.torch = torch
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        chunks = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(
                    texts[i : i + self.batch_size],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                chunks.append(self.model(**enc).logits.float().cpu().numpy())
        return _softmax(np.concatenate(chunks))


class OnnxPredictor:
    name = "onnx"

    def __init__(self, path: str | Path, max_length: int = 256, batch_size: int = 32, quantized: bool = True):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        path = Path(path)
        model_file = path / (
            "model.int8.onnx" if quantized and (path / "model.int8.onnx").exists() else "model.onnx"
        )
        self.name = f"onnx:{model_file.stem}"
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(
                texts[i : i + self.batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="np",
            )
            feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            chunks.append(self.session.run(None, feeds)[0])
        return _softmax(np.concatenate(chunks))


def load_predictor(kind: str, path: str | Path) -> Predictor:
    kinds = {"baseline": BaselinePredictor, "transformer": TransformerPredictor, "onnx": OnnxPredictor}
    if kind not in kinds:
        raise ValueError(f"unknown predictor kind {kind!r}, expected one of {sorted(kinds)}")
    return kinds[kind](path)


@dataclass
class Prediction:
    label: str
    label_en: str
    score: float
    script: str
    top_k: list[dict] = field(default_factory=list)


def classify(predictor: Predictor, texts: list[str], top_k: int = 3) -> list[Prediction]:
    """Normalize (incl. Cyrillic -> Latin), predict and return the top-k labels per text."""
    normalized = [normalize(t) for t in texts]
    proba = predictor.predict_proba(normalized)
    results = []
    for raw, p in zip(texts, proba, strict=True):
        order = np.argsort(p)[::-1][:top_k]
        top = [
            {"label": LABELS[i], "label_en": LABELS_EN[LABELS[i]], "score": round(float(p[i]), 4)}
            for i in order
        ]
        results.append(
            Prediction(
                label=top[0]["label"],
                label_en=top[0]["label_en"],
                score=top[0]["score"],
                script=detect_script(raw),
                top_k=top,
            )
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", nargs="+")
    p.add_argument("--kind", default="onnx", choices=["baseline", "transformer", "onnx"])
    p.add_argument("--path", default="models/onnx")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()
    predictor = load_predictor(args.kind, args.path)
    for pred in classify(predictor, args.text, args.top_k):
        print(json.dumps(pred.__dict__, ensure_ascii=False))


if __name__ == "__main__":
    main()
