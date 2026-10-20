
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml_pipeline import build_preprocessor
from src.train import prepare_train_test_data


def test_train_test_split():
    X_train, X_test, y_train, y_test = prepare_train_test_data()

    assert len(X_train) == 5634
    assert len(X_test) == 1409
    assert len(y_train) == 5634
    assert len(y_test) == 1409


def test_train_test_no_overlap():
    X_train, X_test, _, _ = prepare_train_test_data()

    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_preprocessing_removes_missing_values():
    X_train, _, _, _ = prepare_train_test_data()

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_train)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert not np.isnan(transformed).any()


def test_model_can_predict():
    X_train, X_test, y_train, _ = prepare_train_test_data()

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test.iloc[:5])

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})


def test_model_predicts_probabilities():
    X_train, X_test, y_train, _ = prepare_train_test_data()

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

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test.iloc[:5])

    assert probabilities.shape == (5, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)
