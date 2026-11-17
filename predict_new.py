
import pandas as pd
import numpy as np
import joblib
import os
import argparse

# Load saved model and supporting information
MODEL_PATH = "models/churn_model.pkl"
THRESHOLD_PATH = "models/churn_threshold.pkl"
MEDIAN_PATH = "models/total_charges_median.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)
total_charges_median = joblib.load(MEDIAN_PATH)


def create_features(df):
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Handle missing values using training median
    df["TotalCharges"] = df["TotalCharges"].fillna(
        total_charges_median
    )

    # Convert tenure to numeric
    df["tenure"] = pd.to_numeric(
        df["tenure"], errors="coerce"
    )

    if df["tenure"].isna().any():
        raise ValueError("Missing or invalid tenure values.")

    # Feature 1: TotalServices
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (
        df[service_columns].eq("Yes").sum(axis=1)
    )

    # Feature 2: TenureGroup
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["New", "Medium", "Long", "Very Long"]
    )

    # Feature 3: IsMonthToMonth
    df["IsMonthToMonth"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    # Feature 4: AverageMonthlySpend
    df["AverageMonthlySpend"] = (
        df["TotalCharges"] /
        df["tenure"].replace(0, np.nan)
    ).fillna(0)

    # Feature 5: HasFamily
    df["HasFamily"] = (
        (df["Partner"] == "Yes") &
        (df["Dependents"] == "Yes")
    ).astype(int)

    return df


def predict_customers(input_file):

    # Read new customers
    df = pd.read_csv(input_file)

    # Preserve customer IDs if provided
    customer_ids = (
        df["customerID"].copy()
        if "customerID" in df.columns
        else pd.Series(
            range(1, len(df) + 1),
            name="CustomerNumber"
        )
    )

    # Remove target and identifier columns
    df = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    # Required original customer columns
    required_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Create engineered features
    df = create_features(df)

    # Predict churn probabilities
    probabilities = model.predict_proba(df)[:, 1]

    # Apply business threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Create results
    results = pd.DataFrame({
        "customerID": customer_ids.values,
        "Churn_Probability": probabilities,
        "Predicted_Churn": np.where(
            predictions == 1, "Yes", "No"
        )
    })

    # Save predictions
    results.to_csv("predictions.csv", index=False)

    print("\nPredictions completed successfully!")
    print("Selected threshold:", round(threshold, 3))
    print("Total customers:", len(results))
    print("Predicted churners:", predictions.sum())
    print("Predicted non-churners:", (predictions == 0).sum())
    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))
    print("\nSaved: predictions.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction"
    )

    parser.add_argument(
        "input_file",
        help="Path to new customers CSV file"
    )

    args = parser.parse_args()

    predict_customers(args.input_file)
