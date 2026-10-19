
import pandas as pd

from src.data_preprocessing import clean_data, load_data


def test_load_data():
    df = load_data()

    assert not df.empty
    assert "Churn" in df.columns
    assert "customerID" in df.columns


def test_clean_data_removes_customer_id():
    df = load_data()
    cleaned = clean_data(df)

    assert "customerID" not in cleaned.columns


def test_churn_is_numeric():
    df = load_data()
    cleaned = clean_data(df)

    assert set(cleaned["Churn"].unique()) == {0, 1}


def test_total_charges_is_numeric():
    df = load_data()
    cleaned = clean_data(df)

    assert pd.api.types.is_numeric_dtype(
        cleaned["TotalCharges"]
    )


def test_missing_total_charges():
    df = load_data()
    cleaned = clean_data(df)

    assert cleaned["TotalCharges"].isna().sum() == 11


def test_no_duplicate_customer_ids():
    df = load_data()

    # Add a duplicate customer to verify removal
    duplicate = df.iloc[[0]]
    df_with_duplicate = pd.concat(
        [df, duplicate],
        ignore_index=True
    )

    cleaned = clean_data(df_with_duplicate)

    assert len(cleaned) == len(df)
