import pandas as pd

def run_eda(data_path="data.csv"):
    print("--- Loading Data for EDA ---")
    df = pd.read_csv(data_path, sep=',')

    # --- ADD THIS LINE TO DEBUG ---
    print("Actual columns found in CSV:", df.columns.tolist())
    # ------------------------------

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Churn Rate Distribution ---")
    print(df['Churn'].value_counts(normalize=True) * 100)

    print("\n--- Average Tenure and Charges by Churn ---")
    summary = df.groupby('Churn')[['Tenure', 'MonthlyCharges']].mean()
    print(summary)

    print("\n--- Churn by Contract Type ---")
    print(pd.crosstab(df['ContractType'], df['Churn'], normalize='index') * 100)

if __name__ == "__main__":
    run_eda()