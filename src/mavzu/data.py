"""Download, normalize, subsample and cache the dataset as parquet.

    python -m mavzu.data --out data/processed --max-chars 3000

The raw HF dataset is ~600 MB; processed parquet files are much smaller because each
article is truncated to its beginning (headline + lead carry the topic signal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mavzu.labels import LABELS
from mavzu.text import fingerprint, normalize

DATASET_ID = "murodbek/uz-text-classification"
SPLITS = ("train", "validation", "test")


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Sample n rows preserving the label distribution (every class keeps >= 1 row)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = [
        g.sample(n=max(1, round(len(g) * frac)), random_state=seed) for _, g in df.groupby("label", sort=True)
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_split(data_dir: str | Path, split: str, n: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load a processed split, optionally a stratified subsample of it."""
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    return stratified_sample(df, n, seed)


def prepare(out_dir: Path, max_chars: int, dedup: bool) -> dict:
    from datasets import load_dataset  # heavy import, only needed here

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_ID)

    names = ds["train"].features["label"].names
    if list(names) != LABELS:
        raise ValueError(f"Label set changed upstream: {names}")

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        df = ds[split].to_pandas()
        df["text"] = df["text"].str.slice(0, max_chars).map(normalize)
        df = df[df["text"].str.len() > 20].reset_index(drop=True)
        df["fp"] = df["text"].map(fingerprint)
        frames[split] = df

    stats: dict = {"dataset": DATASET_ID, "max_chars": max_chars, "splits": {}}

    # Leakage audit: the same article is often republished by several outlets.
    train_fps = set(frames["train"]["fp"])
    for split in ("validation", "test"):
        overlap = frames[split]["fp"].isin(train_fps)
        stats[f"{split}_overlap_with_train"] = int(overlap.sum())
        if dedup:
            frames[split] = frames[split][~overlap].reset_index(drop=True)
    stats["train_internal_duplicates"] = int(frames["train"]["fp"].duplicated().sum())
    if dedup:
        frames["train"] = frames["train"].drop_duplicates("fp").reset_index(drop=True)

    for split, df in frames.items():
        df[["text", "label"]].to_parquet(out_dir / f"{split}.parquet", index=False)
        counts = df["label"].value_counts().sort_index()
        stats["splits"][split] = {
            "rows": len(df),
            "per_class": {LABELS[i]: int(c) for i, c in counts.items()},
        }

    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--max-chars", type=int, default=3000)
    p.add_argument("--no-dedup", action="store_true", help="keep train/test duplicates")
    args = p.parse_args()
    stats = prepare(args.out, args.max_chars, dedup=not args.no_dedup)
    print(json.dumps({k: v for k, v in stats.items() if k != "splits"}, indent=2))
    for split, s in stats["splits"].items():
        print(f"{split:<11} {s['rows']:>8,} rows")


if __name__ == "__main__":
    main()
