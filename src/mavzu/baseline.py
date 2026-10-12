"""TF-IDF (word + char n-grams) + calibrated linear SVM baseline.

    python -m mavzu.baseline --config configs/baseline.yaml

Char n-grams matter for Uzbek: it is agglutinative, so one stem appears with dozens of
suffixes (maktab, maktabda, maktablarimizdan ...) and word-level features fragment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from mavzu.data import load_split
from mavzu.metrics import classification_metrics

DEFAULTS = {
    "data_dir": "data/processed",
    "output_dir": "models/baseline",
    "max_train_samples": 100_000,
    "seed": 42,
    "word_max_features": 200_000,
    "char_max_features": 300_000,
    "word_ngram": [1, 2],
    "char_ngram": [2, 5],
    "min_df": 3,
    "C": 0.5,
    "calibration_cv": 3,
}


def build_pipeline(cfg: dict) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=tuple(cfg["word_ngram"]),
                    max_features=cfg["word_max_features"],
                    min_df=cfg["min_df"],
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=tuple(cfg["char_ngram"]),
                    max_features=cfg["char_max_features"],
                    min_df=cfg["min_df"],
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    clf = CalibratedClassifierCV(
        LinearSVC(C=cfg["C"], class_weight="balanced"),
        cv=cfg["calibration_cv"],
        method="sigmoid",
    )
    return Pipeline([("features", features), ("clf", clf)])


def train(cfg: dict) -> dict:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    train_df = load_split(cfg["data_dir"], "train", cfg["max_train_samples"], cfg["seed"])
    val_df = load_split(cfg["data_dir"], "validation")
    print(f"train={len(train_df):,} validation={len(val_df):,}")

    pipe = build_pipeline(cfg)
    t0 = time.perf_counter()
    pipe.fit(train_df["text"].tolist(), train_df["label"].to_numpy())
    fit_s = time.perf_counter() - t0

    proba = pipe.predict_proba(val_df["text"].tolist())
    metrics = classification_metrics(val_df["label"].to_numpy(), proba.argmax(1))
    metrics["fit_seconds"] = round(fit_s, 1)
    metrics["train_samples"] = len(train_df)

    joblib.dump(pipe, out / "model.joblib", compress=3)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "val_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"val accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = p.parse_args()
    cfg = {**DEFAULTS, **(yaml.safe_load(args.config.read_text()) or {})}
    train(cfg)


if __name__ == "__main__":
    main()
