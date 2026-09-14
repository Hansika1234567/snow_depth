"""
train.py
Full training pipeline for Snow Depth Estimation using XGBoost.

Usage:
    python src/train.py
"""

import os
import sys

# Allow imports from src/ when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.preprocessing import preprocess
from src.feature_engineering import add_engineered_features
from src.evaluate import compute_metrics, print_metrics, plot_actual_vs_predicted, plot_feature_importance

DATA_PATH = "data/final_dataset_without_modis_with_features.csv"
TARGET_COLUMN = "SnowDepth"


def train():
    # --- Load and preprocess ---
    print("Loading and preprocessing data...")
    df = preprocess(DATA_PATH)

    # --- Feature engineering ---
    print("Applying feature engineering...")
    df = add_engineered_features(df)

    # --- Separate features and target ---
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {X.shape[1]}")

    # --- Train/test split (80/20, random_state=42) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Train XGBoost model ---
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # --- Predict ---
    y_pred = model.predict(X_test)

    # --- Evaluate ---
    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics)

    # --- Save plots ---
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(model, list(X.columns))

    # --- Sample predictions ---
    print("\nSample Predictions (first 5):")
    for actual, predicted in zip(y_test.values[:5], y_pred[:5]):
        print(f"  Actual: {actual:.2f}  |  Predicted: {predicted:.2f}")

    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    train()
