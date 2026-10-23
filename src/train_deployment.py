
from src.train import (
    train_logistic_regression,
    save_model,
)


def main():
    print("Training deployment model...")

    model = train_logistic_regression()

    model_path = save_model(
        model,
        "logistic_regression.joblib",
    )

    print("Deployment model ready!")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
