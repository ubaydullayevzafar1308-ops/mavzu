import json

import numpy as np

from mavzu.data import load_split, stratified_sample
from mavzu.labels import LABELS
from mavzu.metrics import classification_metrics, confusion, top_confusions
from mavzu.predict import BaselinePredictor, classify
from mavzu.text import latin_to_cyrillic


def test_stratified_sample_keeps_every_class(data_dir):
    df = load_split(data_dir, "train")
    small = stratified_sample(df, 30, seed=0)
    assert set(small["label"]) == set(range(len(LABELS)))
    assert len(small) <= 30 + len(LABELS)


def test_baseline_learns_synthetic_topics(baseline_dir):
    metrics = json.loads((baseline_dir / "val_metrics.json").read_text())
    assert metrics["macro_f1"] > 0.8


def test_predictor_output_shape_and_probabilities(baseline_dir):
    proba = BaselinePredictor(baseline_dir).predict_proba(["futbol gol chempionat", "bank soliq narx"])
    assert proba.shape == (2, len(LABELS))
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-4)


def test_classify_handles_cyrillic_input(baseline_dir):
    predictor = BaselinePredictor(baseline_dir)
    latin = "futbol terma jamoa o‘yin chempionat gol"
    cyr = latin_to_cyrillic(latin)
    out_lat, out_cyr = classify(predictor, [latin, cyr])
    assert out_lat.label == out_cyr.label == "Sport"
    assert out_cyr.script == "cyrillic"
    assert len(out_cyr.top_k) == 3


def test_metrics_helpers():
    y = np.array([0, 0, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 2])
    m = classification_metrics(y, p)
    assert 0 < m["macro_f1"] < 1
    tops = top_confusions(confusion(y, p))
    assert tops[0] == {"true": LABELS[0], "pred": LABELS[1], "rate": 0.5}
