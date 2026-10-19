
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"


def load_data():
    """Load the original customer churn dataset."""
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    """Clean customer churn data without modifying the original."""

    df = df.copy()

    # Convert TotalCharges from text to numeric values
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with missing target values
    df = df.dropna(subset=["Churn"])

    # Remove duplicate customer records
    df = df.drop_duplicates(subset=["customerID"])

    # Convert target variable to numeric
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Remove unique customer identifier
    df = df.drop(columns=["customerID"])

    return df


if __name__ == "__main__":
    raw_df = load_data()
    clean_df = clean_data(raw_df)

    print("Original dataset shape:", raw_df.shape)
    print("Cleaned dataset shape:", clean_df.shape)

    print("\nMissing values after basic cleaning:")
    print(clean_df.isnull().sum())

    print("\nTarget distribution:")
    print(clean_df["Churn"].value_counts())
