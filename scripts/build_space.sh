#!/usr/bin/env bash
# Assemble a self-contained Gradio Space in build/space, then upload it with:
#   hf upload <your-username>/mavzu build/space . --repo-type space
set -euo pipefail
OUT=build/space
rm -rf "$OUT" && mkdir -p "$OUT/models"
cp app.py "$OUT/"
cp -r src "$OUT/"
mkdir -p "$OUT/models/onnx"
cp models/onnx/model.int8.onnx models/onnx/*.json models/onnx/*.txt "$OUT/models/onnx/" 2>/dev/null || true
cat > "$OUT/requirements.txt" <<'REQ'
numpy
pandas
scikit-learn
pyyaml
onnxruntime
transformers
REQ
cat > "$OUT/README.md" <<'MD'
---
title: Mavzu — Uzbek news topic classifier
emoji: 🇺🇿
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---
Classifies Uzbek news text (Latin or Cyrillic) into 15 topics. Fine-tuned TahrirchiBERT, INT8 ONNX.
MD
echo "Space assembled in $OUT ($(du -sh "$OUT" | cut -f1))"
