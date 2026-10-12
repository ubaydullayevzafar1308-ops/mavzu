"""Tiny synthetic dataset so tests run in seconds without downloading anything."""

import random

import pandas as pd
import pytest

from mavzu.labels import LABELS

# A few topic words per class; enough for a TF-IDF model to separate them.
VOCAB = {
    "Avto": "avtomobil mashina yo‘l haydovchi benzin",
    "Ayollar": "ayollar onalar go‘zallik oila kiyim",
    "Dunyo": "xalqaro davlatlar prezident Yevropa Osiyo",
    "Foto": "surat fotograf lavha kadr galereya",
    "Iqtisodiyot": "bank inflyatsiya narx eksport soliq",
    "Jamiyat": "fuqarolar mahalla aholi jamoatchilik muammo",
    "Jinoyat": "jinoyat sud ichki ishlar firibgarlik hibs",
    "Madaniyat": "teatr kino musiqa san’at kitob",
    "O‘zbekiston": "viloyat hokim tuman respublika vazirlik",
    "Pazandachilik": "palov retsept guruch go‘sht taom",
    "Qonunchilik": "qonun farmon qaror kodeks modda",
    "Salomatlik": "shifokor kasallik emlash sog‘liq dori",
    "Siyosat": "parlament saylov partiya deputat siyosat",
    "Sport": "futbol terma jamoa o‘yin chempionat gol",
    "Texnologiya": "smartfon internet dastur sun’iy intellekt kompyuter",
}
FILLER = ["va", "bilan", "uchun", "haqida", "bugun", "yangi", "katta"]


def make_split(n_per_class: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label_id, name in enumerate(LABELS):
        words = VOCAB[name].split()
        for _ in range(n_per_class):
            text = " ".join(rng.choice(words + FILLER) for _ in range(14))
            rows.append({"text": text, "label": label_id})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    for split, n, seed in (("train", 12, 0), ("validation", 4, 1), ("test", 4, 2)):
        make_split(n, seed).to_parquet(d / f"{split}.parquet", index=False)
    return d


@pytest.fixture(scope="session")
def baseline_dir(data_dir, tmp_path_factory):
    from mavzu.baseline import DEFAULTS, train

    out = tmp_path_factory.mktemp("baseline")
    cfg = {
        **DEFAULTS,
        "data_dir": str(data_dir),
        "output_dir": str(out),
        "max_train_samples": None,
        "min_df": 1,
        "word_max_features": 5000,
        "char_max_features": 5000,
        "calibration_cv": 2,
    }
    train(cfg)
    return out
