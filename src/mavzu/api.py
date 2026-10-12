"""FastAPI service.

    MODEL_KIND=onnx MODEL_PATH=models/onnx uvicorn mavzu.api:app --port 8000

Endpoints: GET /health, GET /labels, POST /predict, POST /predict/batch
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mavzu.labels import LABELS, LABELS_EN
from mavzu.predict import Predictor, classify, load_predictor

MAX_CHARS = 20_000
MAX_BATCH = 64


class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=MAX_CHARS, examples=["Ўзбекистон терма жамоаси ғалаба қозонди"]
    )
    top_k: int = Field(3, ge=1, le=len(LABELS))


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    top_k: int = Field(3, ge=1, le=len(LABELS))


class LabelScore(BaseModel):
    label: str
    label_en: str
    score: float


class PredictResponse(BaseModel):
    label: str
    label_en: str
    score: float
    script: str
    top_k: list[LabelScore]


class BatchResponse(BaseModel):
    predictions: list[PredictResponse]
    model: str
    latency_ms: float


def create_app(predictor: Predictor | None = None) -> FastAPI:
    state: dict = {"predictor": predictor}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["predictor"] is None:
            kind = os.getenv("MODEL_KIND", "onnx")
            path = os.getenv("MODEL_PATH", "models/onnx")
            state["predictor"] = load_predictor(kind, path)
        yield

    app = FastAPI(
        title="Mavzu — Uzbek news topic classifier",
        version="0.1.0",
        description="Classifies Uzbek text (Latin or Cyrillic) into 15 news topics.",
        lifespan=lifespan,
    )

    def get_predictor() -> Predictor:
        p = state["predictor"]
        if p is None:
            raise HTTPException(503, "model is not loaded")
        return p

    @app.get("/health")
    def health() -> dict:
        p = state["predictor"]
        return {"status": "ok" if p else "loading", "model": getattr(p, "name", None)}

    @app.get("/labels")
    def labels() -> list[dict]:
        return [{"id": i, "label": name, "label_en": LABELS_EN[name]} for i, name in enumerate(LABELS)]

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        if not req.text.strip():
            raise HTTPException(422, "text is empty")
        pred = classify(get_predictor(), [req.text], req.top_k)[0]
        return PredictResponse(**pred.__dict__)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest) -> BatchResponse:
        if any(len(t) > MAX_CHARS for t in req.texts):
            raise HTTPException(422, f"each text must be at most {MAX_CHARS} characters")
        predictor = get_predictor()
        t0 = time.perf_counter()
        preds = classify(predictor, req.texts, req.top_k)
        return BatchResponse(
            predictions=[PredictResponse(**p.__dict__) for p in preds],
            model=predictor.name,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    return app


app = create_app()
