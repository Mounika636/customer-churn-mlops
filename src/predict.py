
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression.joblib"
)


def load_model():
    """Load the previously trained ML model."""
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data: dict):
    """Predict churn for one customer."""

    model = load_model()

    # Convert customer data into a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Generate prediction and probability
    prediction = int(model.predict(customer_df)[0])

    churn_probability = float(
        model.predict_proba(customer_df)[0][1]
    )

    return {
        "prediction": prediction,
        "churn_probability": round(
            churn_probability, 4
        ),
        "churn_label": (
            "Yes" if prediction == 1 else "No"
        ),
    }


if __name__ == "__main__":
    from src.data_preprocessing import load_data, clean_data

    df = clean_data(load_data())

    # Take one sample customer without its target label
    sample_customer = df.drop(
        columns=["Churn"]
    ).iloc[0].to_dict()

    result = predict_churn(sample_customer)

    print("\n========== CUSTOMER CHURN PREDICTION ==========")
    print("Predicted churn:", result["churn_label"])
    print(
        "Churn probability:",
        f"{result['churn_probability']:.2%}"
    )
