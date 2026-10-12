"""Fine-tune TahrirchiBERT (Uzbek BERT) for news topic classification.

    python -m mavzu.train_transformer --config configs/transformer.yaml

Designed to run on a MacBook (Apple Silicon, MPS backend): a stratified subsample of
the training set, dynamic padding and a modest max_length keep an epoch practical.
On a CUDA GPU just raise max_train_samples / batch size in the config.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from mavzu.data import load_split
from mavzu.labels import ID2LABEL, LABEL2ID, LABELS
from mavzu.metrics import classification_metrics

# Lets the few ops MPS does not implement fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULTS = {
    "model_name": "tahrirchi/tahrirchi-bert-small",
    "data_dir": "data/processed",
    "output_dir": "models/transformer",
    "max_train_samples": 60_000,
    "max_eval_samples": 10_000,
    "max_length": 256,
    "epochs": 2,
    "learning_rate": 5.0e-5,
    "batch_size": 16,
    "grad_accum": 2,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "label_smoothing": 0.05,
    "eval_steps": 500,
    "early_stopping_patience": 3,
    "seed": 42,
    "report_to": "none",  # "tensorboard" or "wandb"
}


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def last_checkpoint(ckpt_dir: Path) -> str | None:
    """Most recently written checkpoint in ckpt_dir, or None.

    Picked by write time, not step number: the directory can still hold higher-numbered
    checkpoints of an earlier, shorter run.
    """
    states = sorted(ckpt_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: p.stat().st_mtime)
    return str(states[-1].parent) if states else None


def train(cfg: dict, resume: bool = False) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg["seed"])
    device = pick_device()
    print(f"device: {device}")

    out = Path(cfg["output_dir"])
    resume_from = last_checkpoint(out / "checkpoints") if resume else None
    if resume:
        print(f"resuming from {resume_from}" if resume_from else "no checkpoint found, starting from scratch")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_df = load_split(cfg["data_dir"], "train", cfg["max_train_samples"], cfg["seed"])
    val_df = load_split(cfg["data_dir"], "validation", cfg["max_eval_samples"], cfg["seed"])
    print(f"train={len(train_df):,} validation={len(val_df):,}")

    def tokenize(batch: dict) -> dict:
        return tok(batch["text"], truncation=True, max_length=cfg["max_length"])

    cols = ["text", "label"]
    train_ds = Dataset.from_pandas(train_df[cols]).map(tokenize, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_pandas(val_df[cols]).map(tokenize, batched=True, remove_columns=["text"])

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        m = classification_metrics(labels, np.argmax(logits, axis=-1))
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["grad_accum"],
        warmup_steps=cfg["warmup_ratio"],  # transformers 5 dropped warmup_ratio; a float < 1 is a ratio
        weight_decay=cfg["weight_decay"],
        label_smoothing_factor=cfg["label_smoothing"],
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=device == "cuda",  # mixed precision is unreliable on MPS, keep fp32 there
        dataloader_num_workers=0,  # multiprocessing + MPS on macOS is a common source of hangs
        report_to=cfg["report_to"],
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(cfg["early_stopping_patience"])],
    )

    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_from)
    train_s = time.perf_counter() - t0

    final_dir = out / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))

    eval_metrics = trainer.evaluate()
    summary = {
        "model_name": cfg["model_name"],
        "device": device,
        "train_samples": len(train_df),
        "train_minutes": round(train_s / 60, 1),  # this session only when resumed
        "resumed_from": resume_from,
        "val_accuracy": eval_metrics.get("eval_accuracy"),
        "val_macro_f1": eval_metrics.get("eval_macro_f1"),
        "torch": torch.__version__,
    }
    (final_dir / "train_config.json").write_text(json.dumps(cfg, indent=2))
    (final_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/transformer.yaml"))
    p.add_argument("--max-train-samples", type=int, help="override config (quick smoke runs)")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in output_dir")
    args = p.parse_args()
    cfg = {**DEFAULTS, **(yaml.safe_load(args.config.read_text()) or {})}
    if args.max_train_samples:
        cfg["max_train_samples"] = args.max_train_samples
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
