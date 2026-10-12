import pytest
from fastapi.testclient import TestClient

from mavzu.api import MAX_BATCH, create_app
from mavzu.labels import LABELS
from mavzu.predict import BaselinePredictor


@pytest.fixture(scope="module")
def client(baseline_dir):
    with TestClient(create_app(BaselinePredictor(baseline_dir))) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "baseline"}


def test_labels(client):
    labels = client.get("/labels").json()
    assert [x["label"] for x in labels] == LABELS


def test_predict_cyrillic(client):
    r = client.post("/predict", json={"text": "Шифокор касаллик эмлаш дори", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Salomatlik"
    assert body["script"] == "cyrillic"
    assert len(body["top_k"]) == 2


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"texts": ["palov retsept guruch", "parlament saylov partiya"]})
    assert r.status_code == 200
    assert [p["label"] for p in r.json()["predictions"]] == ["Pazandachilik", "Siyosat"]


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "   "}, {"text": "salom", "top_k": 0}, {"texts": ["a"] * (MAX_BATCH + 1)}],
)
def test_validation_errors(client, payload):
    path = "/predict/batch" if "texts" in payload else "/predict"
    assert client.post(path, json=payload).status_code == 422
