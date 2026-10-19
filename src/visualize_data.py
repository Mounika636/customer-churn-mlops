
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")


# Chart 1: Overall churn distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn", hue="Churn", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
plt.close()


# Chart 2: Churn rate by contract type
contract_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
      .groupby("Contract")["ChurnFlag"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
      .reset_index()
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=contract_churn,
    x="Contract",
    y="ChurnFlag",
    hue="Contract",
    legend=False,
)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Contract Type")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
plt.close()


# Chart 3: Customer tenure by churn status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tenure_vs_churn.png", dpi=150)
plt.close()


# Chart 4: Monthly charges by churn status
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_charges_vs_churn.png", dpi=150)
plt.close()


print("Successfully created 4 visualizations!")
print(f"Saved in: {OUTPUT_DIR}")
