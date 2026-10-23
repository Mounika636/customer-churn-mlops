
import mlflow
import mlflow.sklearn

from src.data_preprocessing import load_data, clean_data


MLFLOW_URI = "http://127.0.0.1:5000"

LOGISTIC_RUN_ID = "bc681c6c782740bc96db506fb1030a5d"


def test_logged_model():
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Find the model logged by this run
    client = mlflow.tracking.MlflowClient()

    models = client.search_logged_models(
        experiment_ids=["1"],
        filter_string=f"source_run_id = '{LOGISTIC_RUN_ID}'",
    )

    if not models:
        raise RuntimeError("No logged model found for this run")

    model_id = models[0].model_id
    model_uri = f"models:/{model_id}"

    print("Loading MLflow model:", model_uri)

    # These trusted types belong to our own trained pipeline
  
    model = mlflow.sklearn.load_model(model_uri)


    df = clean_data(load_data())
    sample_customer = df.drop(columns=["Churn"]).iloc[[0]]

    prediction = int(model.predict(sample_customer)[0])
    probability = float(
        model.predict_proba(sample_customer)[0][1]
    )

    print("\n========== MLFLOW MODEL PREDICTION ==========")
    print("Predicted churn:", "Yes" if prediction else "No")
    print(f"Churn probability: {probability:.2%}")


if __name__ == "__main__":
    test_logged_model()
