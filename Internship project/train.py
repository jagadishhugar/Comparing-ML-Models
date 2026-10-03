import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


def load_and_preprocess(data_path="data.csv"):
    df = pd.read_csv(data_path)

    # Drop CustomerID as it's just an identifier
    X = df.drop(columns=['CustomerID', 'Churn'])
    y = df['Churn']

    # Define categorical and numerical features
    categorical_cols = ['ContractType', 'TechSupport']
    numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)
    return train_test_split(X_processed, y, test_size=0.3, random_state=42, stratify=y)


def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("---Model Evaluation---")
    print(f"Accuracy: {acc:.2f}")
    print(f"Recall:   {rec:.2f}")
    print(f"ROC-AUC:  {auc:.2f}\n")


def main():
    # 1. Split Data
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # 2. Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }

    # 3. Train and Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name)


if __name__ == "__main__":
    main()