import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.predict import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model once when the API starts.
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using machine learning.",
    version="1.0.0",
    lifespan=lifespan,
)


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)



@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "model": "Logistic Regression",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None,
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        model = app.state.model

        prediction = int(
            model.predict(customer_df)[0]
        )

        probability = float(
            model.predict_proba(customer_df)[0][1]
        )

        return {
            "prediction": prediction,
            "churn_label": (
                "Yes" if prediction == 1 else "No"
            ),
            "churn_probability": round(probability, 4),
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
