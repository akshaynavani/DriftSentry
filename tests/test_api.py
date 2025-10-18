import pytest
from fastapi.testclient import TestClient

from driftsentry.api.main import create_app
from driftsentry.api.service import InferenceService
from driftsentry.api.settings import Settings
from driftsentry.config import DataConfig
from driftsentry.data.synthetic import generate_transactions

API_COLUMNS = ["txn_id", "account_id", "timestamp", "amount", "txn_type", "channel"]


def _payload(df):
    records = df[API_COLUMNS].copy()
    records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return records.to_dict("records")


@pytest.fixture(scope="module")
def client(bundle):
    app = create_app(service=InferenceService(bundle), settings=Settings(max_transactions=5000))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client, bundle):
    body = client.get("/model/info").json()
    assert body["threshold"] == pytest.approx(bundle.threshold)
    assert body["seq_len"] == bundle.window_cfg.seq_len
    assert len(body["features"]) == bundle.model_cfg.n_features


def test_score_returns_one_entry_per_account(client, transactions):
    accounts = transactions["account_id"].unique()[:3]
    df = transactions[transactions["account_id"].isin(accounts)]
    r = client.post("/score", json={"transactions": _payload(df), "include_windows": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {a["account_id"] for a in body["accounts"]} == set(accounts)
    for acc in body["accounts"]:
        assert acc["max_error"] >= acc["latest_error"] or acc["n_windows"] == 1
        assert len(acc["top_features"]) == 3
        assert len(acc["windows"]) == acc["n_windows"]


def test_score_validates_input(client):
    bad = {"transactions": [{"account_id": "a", "timestamp": "2025-01-01T00:00:00",
                             "amount": -5, "txn_type": "deposit", "channel": "atm"}]}
    assert client.post("/score", json=bad).status_code == 422
    assert client.post("/score", json={"transactions": []}).status_code == 422


def test_drift_detects_shifted_population(client):
    shifted = generate_transactions(
        DataConfig(n_accounts=25, days=30, seed=99), amount_shift=2.5, anomalies=False
    )
    r = client.post("/drift", json={"transactions": _payload(shifted)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["drift_detected"]
    assert "log_amount" in body["drifted_features"]


def test_request_size_limit(bundle):
    app = create_app(service=InferenceService(bundle), settings=Settings(max_transactions=1))
    with TestClient(app) as c:
        txn = {"account_id": "a", "timestamp": "2025-01-01T00:00:00", "amount": 5,
               "txn_type": "deposit", "channel": "atm"}
        assert c.post("/score", json={"transactions": [txn, txn]}).status_code == 413


def test_service_unavailable_without_model(tmp_path):
    app = create_app(settings=Settings(artifacts_dir=str(tmp_path)))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/model/info").status_code == 503
