---
language: uz
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: tahrirchi/tahrirchi-bert-small
datasets:
  - murodbek/uz-text-classification
metrics:
  - f1
  - accuracy
tags:
  - uzbek
  - news
  - topic-classification
  - onnx
---

# Mavzu — Uzbek news topic classifier

Fine-tuned [TahrirchiBERT-small](https://huggingface.co/tahrirchi/tahrirchi-bert-small) that
classifies Uzbek news text into 15 topics. Includes an INT8 ONNX export for CPU inference.

Code, training details and evaluation: https://github.com/ubaydullayevzafar1308-ops/mavzu

## Labels

Avto, Ayollar, Dunyo, Foto, Iqtisodiyot, Jamiyat, Jinoyat, Madaniyat, O‘zbekiston,
Pazandachilik, Qonunchilik, Salomatlik, Siyosat, Sport, Texnologiya

## Evaluation (deduplicated test split)

| Variant | Macro-F1 | Accuracy |
|---|---:|---:|
| PyTorch fp32 | TBD | TBD |
| ONNX INT8 | TBD | TBD |

## Intended use and limitations

- The model expects **Latin-script** input. Cyrillic text must be transliterated first; use
  `mavzu.text.normalize` from the repository, otherwise quality drops sharply.
- Trained on news articles, so short informal or code-switched (Uzbek-Russian) text is out of domain.
- Labels reflect the section each outlet chose, so neighbouring topics overlap.

## Training

Stratified subsample of the train split, max length 256, AdamW, linear schedule with warm-up,
label smoothing 0.05, early stopping on validation macro-F1. Trained on Apple Silicon (MPS).
Full config: `train_config.json`.
