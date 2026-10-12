.PHONY: setup data baseline train eval export serve demo docker test lint space

PY ?= python

setup:            ## install everything for local development (MacBook / conda)
	$(PY) -m pip install -e ".[train,onnx,serve,demo,dev]"

data:             ## download, normalize, dedup and cache the dataset
	$(PY) -m mavzu.data --out data/processed

baseline:         ## TF-IDF + LinearSVC baseline
	$(PY) -m mavzu.baseline --config configs/baseline.yaml
	$(PY) -m mavzu.evaluate --kind baseline --path models/baseline

smoke:            ## 2k-sample transformer run to check the pipeline end to end
	$(PY) -m mavzu.train_transformer --config configs/transformer.yaml --max-train-samples 2000

train:            ## fine-tune TahrirchiBERT
	$(PY) -m mavzu.train_transformer --config configs/transformer.yaml

eval:             ## evaluate the fine-tuned model (PyTorch) and the ONNX export
	$(PY) -m mavzu.evaluate --kind transformer --path models/transformer/final --max-samples 5000
	$(PY) -m mavzu.evaluate --kind onnx --path models/onnx --max-samples 5000

export:           ## ONNX fp32 + INT8 export with parity check
	$(PY) -m mavzu.export_onnx --model models/transformer/final --out models/onnx

serve:
	MODEL_KIND=onnx MODEL_PATH=models/onnx uvicorn mavzu.api:app --reload --port 8000

demo:
	MODEL_KIND=onnx MODEL_PATH=models/onnx $(PY) app.py

docker:
	docker build -t mavzu . && docker run --rm -p 8000:8000 mavzu

space:            ## assemble a Hugging Face Space in build/space
	bash scripts/build_space.sh

test:
	pytest -q

lint:
	ruff check . && ruff format --check .
