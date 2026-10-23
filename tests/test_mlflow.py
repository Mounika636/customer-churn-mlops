
from unittest.mock import patch

from src.mlflow_tracking import (
    EXPERIMENT_NAME,
    MLFLOW_URI,
    track_experiments,
)


def test_mlflow_configuration():
    assert EXPERIMENT_NAME == "Customer Churn Prediction"
    assert MLFLOW_URI == "http://127.0.0.1:5000"


def test_mlflow_experiment_setup():
    # Stop execution after experiment setup so that
    # no training or actual logging occurs.
    with patch(
        "src.mlflow_tracking.mlflow.set_tracking_uri"
    ) as mock_uri, patch(
        "src.mlflow_tracking.mlflow.set_experiment"
    ) as mock_experiment, patch(
        "src.mlflow_tracking.prepare_train_test_data",
        side_effect=RuntimeError("Stop before training"),
    ):

        try:
            track_experiments()
        except RuntimeError as error:
            assert str(error) == "Stop before training"

        mock_uri.assert_called_once_with(MLFLOW_URI)
        mock_experiment.assert_called_once_with(
            EXPERIMENT_NAME
        )
