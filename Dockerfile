# Serves the INT8 ONNX model with FastAPI. No PyTorch in the image.
#   make export && docker build -t mavzu . && docker run -p 8000:8000 mavzu
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_KIND=onnx \
    MODEL_PATH=/app/models/onnx

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,onnx]"

COPY models/onnx ./models/onnx

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "mavzu.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
