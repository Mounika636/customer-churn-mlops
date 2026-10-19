
from pathlib import Path
import pandas as pd

# Find the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the dataset
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== CHURN DISTRIBUTION ==========")
print(df["Churn"].value_counts())

print("\n========== CHURN PERCENTAGES ==========")
print(
    df["Churn"].value_counts(normalize=True) * 100
)

print("\n========== TOTAL CHARGES CHECK ==========")

# Check whether TotalCharges contains invalid numbers
total_charges = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Invalid or missing TotalCharges values:")
print(total_charges.isnull().sum())

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe())
