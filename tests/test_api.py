
import pytest
from fastapi.testclient import TestClient

from src.api import app



from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    mock_model = MagicMock()

    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.3882, 0.6118]]

    with patch("src.api.load_model", return_value=mock_model):
        with TestClient(app) as test_client:
            yield test_client



@pytest.fixture
def valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction(client, valid_customer):
    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1


def test_negative_tenure_rejected(client, valid_customer):
    valid_customer["tenure"] = -5

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_invalid_contract_rejected(client, valid_customer):
    valid_customer["Contract"] = "Unlimited"

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_missing_feature_rejected(client, valid_customer):
    del valid_customer["MonthlyCharges"]

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422


def test_negative_charges_rejected(client, valid_customer):
    valid_customer["MonthlyCharges"] = -100

    response = client.post("/predict", json=valid_customer)

    assert response.status_code == 422
