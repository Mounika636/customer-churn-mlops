
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.ml_pipeline import build_preprocessor
from src.train import prepare_train_test_data


MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Customer Churn Prediction"


def track_experiments():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = prepare_train_test_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for model_name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        with mlflow.start_run(run_name=model_name):

            # Record model settings
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("test_size", 0.2)

            for key, value in classifier.get_params().items():
                if key in [
                    "max_iter",
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                ]:
                    mlflow.log_param(key, value)

            # Train the model
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }

            # Save metrics to MLflow
            mlflow.log_metrics(metrics)

            # Save the entire trained pipeline
            
            
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="churn_model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    ],
                    )

            print(f"\nLogged experiment: {model_name}")

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    track_experiments()
