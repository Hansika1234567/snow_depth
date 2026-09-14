"""
evaluate.py
Evaluation functions: metrics, feature importance plot, actual vs predicted plot.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, and R² for regression evaluation.
    Returns a dictionary with the three metric values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics(metrics):
    """Print evaluation metrics in a readable format."""
    print("=" * 35)
    print("Model Evaluation Results")
    print("=" * 35)
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R²   : {metrics['R2']:.4f}")
    print("=" * 35)


def plot_actual_vs_predicted(y_true, y_pred, save_path="results/actual_vs_predicted.png"):
    """
    Plot actual vs predicted snow depth values and save to file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit")
    plt.xlabel("Actual Snow Depth")
    plt.ylabel("Predicted Snow Depth")
    plt.title("Actual vs Predicted Snow Depth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(model, feature_names, top_n=15, save_path="results/feature_importance.png"):
    """
    Plot the top N feature importances from the trained XGBoost model and save to file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
