"""
train_model.py
---------------
ONE JOB: load the loan dataset, clean it, encode it, train several
classification models, compare them, and save the best one to disk
so app.py can load it later and serve predictions.

Run:
    python train_model.py

Output:
    model.pkl   -> trained model + encoders + column order (used by app.py)
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_PATH = "data/loan_data.csv"
MODEL_PATH = "model.pkl"

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
TARGET_COL = "Loan_Status"


# ---------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------
def load_data():
    """
    Loads the Kaggle/Analytics Vidhya 'Loan Prediction' dataset if present
    at data/loan_data.csv. If the file is missing, a small synthetic
    dataset with the SAME columns is generated so the pipeline still runs
    end to end. Replace data/loan_data.csv with the real dataset for a
    real project.
    """
    if os.path.exists(DATA_PATH):
        print(f"Loading real dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        return df

    print(f"No dataset found at {DATA_PATH}. Generating synthetic data instead.")
    print("Download the real dataset (Kaggle: 'Loan Prediction Problem Dataset')")
    print(f"and place it at {DATA_PATH} to train on real data.\n")

    rng = np.random.default_rng(42)
    n = 600

    df = pd.DataFrame({
        "Gender": rng.choice(["Male", "Female", None], size=n, p=[0.7, 0.27, 0.03]),
        "Married": rng.choice(["Yes", "No", None], size=n, p=[0.6, 0.37, 0.03]),
        "Dependents": rng.choice(["0", "1", "2", "3+", None], size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "Education": rng.choice(["Graduate", "Not Graduate"], size=n, p=[0.78, 0.22]),
        "Self_Employed": rng.choice(["Yes", "No", None], size=n, p=[0.13, 0.8, 0.07]),
        "ApplicantIncome": rng.integers(1500, 20000, size=n).astype(float),
        "CoapplicantIncome": rng.integers(0, 8000, size=n).astype(float),
        "LoanAmount": rng.integers(50, 500, size=n).astype(float),
        "Loan_Amount_Term": rng.choice([360, 180, 120, 300, np.nan], size=n, p=[0.75, 0.1, 0.05, 0.05, 0.05]),
        "Credit_History": rng.choice([1.0, 0.0, np.nan], size=n, p=[0.75, 0.15, 0.1]),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], size=n),
    })

    # Make the target loosely depend on the features so models can learn
    # something real instead of pure noise.
    score = (
        (df["Credit_History"].fillna(0) == 1.0).astype(int) * 3
        + (df["Education"] == "Graduate").astype(int)
        + (df["ApplicantIncome"] + df["CoapplicantIncome"] > 4000).astype(int)
        - (df["LoanAmount"] > 300).astype(int)
        + rng.normal(0, 1, size=n)
    )
    df[TARGET_COL] = np.where(score > 2, "Y", "N")

    # Sprinkle in some missing values to force real cleaning work
    for col in ["LoanAmount", "ApplicantIncome"]:
        idx = rng.choice(n, size=int(n * 0.03), replace=False)
        df.loc[idx, col] = np.nan

    return df


# ---------------------------------------------------------------------
# 2. HANDLE MISSING VALUES
# ---------------------------------------------------------------------
def clean_data(df):
    df = df.copy()

    # Categorical -> fill with the most frequent value (mode)
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    # Numeric -> fill with median (robust to outliers, common for income/loan amount)
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Drop rows only if the target itself is missing
    df = df.dropna(subset=[TARGET_COL])

    return df


# ---------------------------------------------------------------------
# 3. ENCODE CATEGORICAL COLUMNS
# ---------------------------------------------------------------------
def encode_data(df):
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    df[TARGET_COL] = target_encoder.fit_transform(df[TARGET_COL].astype(str))
    encoders[TARGET_COL] = target_encoder

    return df, encoders


# ---------------------------------------------------------------------
# 4. TRAIN + COMPARE MULTIPLE MODELS
# ---------------------------------------------------------------------
def train_and_compare(df):
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []
    trained_models = {}

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        # In this dataset: label 1 = "Y" (approved), label 0 = "N" (rejected)
        # cm layout: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1 score : {f1:.3f}")
        print(f"False Approvals  (rejected app predicted approved) : {fp}")
        print(f"False Rejections (approved app predicted rejected) : {fn}")

        results.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_approvals": int(fp),
            "false_rejections": int(fn),
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by F1 score)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    print("""
Why false approvals and false rejections matter here:
  - A FALSE APPROVAL means the model told the bank to approve a loan
    that should have been rejected -> real financial (credit) risk.
  - A FALSE REJECTION means a genuinely credit-worthy applicant was
    turned down -> lost business and an unfair outcome for the applicant.
  A production loan model usually tunes its decision threshold to reduce
  false approvals first, since they carry direct monetary loss, even if
  that means accepting a few more false rejections.
""")

    return best_model, feature_cols, results_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    df = load_data()
    df = clean_data(df)
    df, encoders = encode_data(df)
    best_model, feature_cols, results_df = train_and_compare(df)

    bundle = {
        "model": best_model,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
