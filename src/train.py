from sklearn.linear_model import LogisticRegression
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

try:
    from src.ml_pipeline import build_preprocessor
except ModuleNotFoundError:
    from ml_pipeline import build_preprocessor
from sklearn.model_selection import train_test_split

try:
    from src.data_preprocessing import load_data, clean_data
except ModuleNotFoundError:
    from data_preprocessing import load_data, clean_data


def prepare_train_test_data():
    # Load and clean the dataset
    df = clean_data(load_data())

    # Separate input features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test



def train_logistic_regression():
    """Train and evaluate a customer churn classifier."""

    X_train, X_test, y_train, y_test = (
        prepare_train_test_data()
    )

    # Combine preprocessing and model training
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Fit only on training data
    model.fit(X_train, y_train)

    # Evaluate on held-out test data
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n========== LOGISTIC REGRESSION RESULTS ==========")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    return model



def train_random_forest():
    """Train and evaluate a Random Forest churn classifier."""

    X_train, X_test, y_train, y_test = (
        prepare_train_test_data()
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Train only on the training dataset
    model.fit(X_train, y_train)

    # Predict customer churn
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n========== RANDOM FOREST RESULTS ==========")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    return model


def save_model(model, filename="logistic_regression.joblib"):
    """Save a trained ML pipeline to disk."""

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / filename

    joblib.dump(model, model_path)

    print(f"\nModel saved successfully: {model_path}")

    return model_path



if __name__ == "__main__":
    logistic_model = train_logistic_regression()
    random_forest_model = train_random_forest()

    save_model(logistic_model, "logistic_regression.joblib")
    save_model(random_forest_model, "random_forest.joblib")