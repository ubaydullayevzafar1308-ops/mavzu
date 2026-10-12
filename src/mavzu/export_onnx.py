"""Export the fine-tuned model to ONNX and quantize it to INT8 for cheap CPU serving.

    python -m mavzu.export_onnx --model models/transformer/final --out models/onnx

Writes model.onnx (fp32), model.int8.onnx (dynamic INT8) and the tokenizer files, then
checks that both ONNX models agree with PyTorch on a sample of validation texts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mavzu.data import load_split


def export(model_dir: Path, out: Path, opset: int = 17) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval().cpu()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits

    dummy = tok(["salom dunyo"], return_tensors="pt")
    if "token_type_ids" not in dummy:
        dummy["token_type_ids"] = torch.zeros_like(dummy["input_ids"])
    args = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])
    dyn = {0: "batch", 1: "seq"}
    onnx_path = out / "model.onnx"
    kwargs = dict(
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": dyn, "attention_mask": dyn, "token_type_ids": dyn, "logits": {0: "batch"}},
        opset_version=opset,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(LogitsOnly(model), args, str(onnx_path), dynamo=False, **kwargs)
        except TypeError:  # torch versions without the `dynamo` flag
            torch.onnx.export(LogitsOnly(model), args, str(onnx_path), **kwargs)

    tok.save_pretrained(str(out))
    model.config.save_pretrained(str(out))
    return onnx_path


def quantize(onnx_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_path = onnx_path.with_name("model.int8.onnx")
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
    return q_path


def parity(model_dir: Path, out: Path, data_dir: str, n: int) -> dict:
    from mavzu.predict import OnnxPredictor, TransformerPredictor

    texts = load_split(data_dir, "validation", n, seed=3)["text"].tolist()
    ref = TransformerPredictor(model_dir).predict_proba(texts)
    report = {"n": len(texts)}
    for quantized in (False, True):
        pred = OnnxPredictor(out, quantized=quantized).predict_proba(texts)
        key = "int8" if quantized else "fp32"
        report[key] = {
            "max_abs_prob_diff": round(float(np.abs(pred - ref).max()), 5),
            "argmax_agreement": round(float((pred.argmax(1) == ref.argmax(1)).mean()), 4),
        }
    for f in ("model.onnx", "model.int8.onnx"):
        report[f"{f}_mb"] = round((out / f).stat().st_size / 2**20, 1)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/transformer/final"))
    p.add_argument("--out", type=Path, default=Path("models/onnx"))
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--parity-n", type=int, default=500)
    args = p.parse_args()

    onnx_path = export(args.model, args.out)
    quantize(onnx_path)
    report = parity(args.model, args.out, args.data_dir, args.parity_n)
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
