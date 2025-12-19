import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns


def get_paths():
    """
    Resolve paths relative to the project root.

    Expected structure:
    - project_root/
        - data/services_weekly.csv
        - results/
        - src/train_model.py  (this file)
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, ".."))

    data_path = os.path.join(project_root, "data", "services_weekly.csv")
    results_dir = os.path.join(project_root, "results")

    os.makedirs(results_dir, exist_ok=True)

    return data_path, results_dir


def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cannot find dataset at: {data_path}\n"
            "Make sure data/services_weekly.csv exists."
        )

    df = pd.read_csv(data_path)
    print("Data loaded. Shape:", df.shape)
    return df


def prepare_features(df: pd.DataFrame):
    """
    Select features and target.
    Adjust column names here if your CSV is different.
    """
    feature_cols = [
        "available_beds",
        "patients_request",
        "patients_admitted",
        "patients_refused",
        "staff_morale",
    ]
    target_col = "patient_satisfaction"

    # Safety check
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    X = df[feature_cols]
    y = df[target_col]

    return X, y, feature_cols


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return model, y_pred, metrics


def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state=42,
):
    base_model = RandomForestRegressor(random_state=random_state)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    print("Fitting Random Forest with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return best_model, y_pred, metrics


def save_metrics(results_dir, metrics_lr, metrics_rf):
    """
    Save metrics to results/metrics.txt
    """
    metrics_path = os.path.join(results_dir, "metrics.txt")
    lines = []
    lines.append("=== Model Performance ===\n")
    lines.append("Linear Regression:\n")
    lines.append(f"  MAE : {metrics_lr['MAE']:.4f}\n")
    lines.append(f"  RMSE: {metrics_lr['RMSE']:.4f}\n")
    lines.append(f"  R2  : {metrics_lr['R2']:.4f}\n\n")

    lines.append("Random Forest:\n")
    lines.append(f"  MAE : {metrics_rf['MAE']:.4f}\n")
    lines.append(f"  RMSE: {metrics_rf['RMSE']:.4f}\n")
    lines.append(f"  R2  : {metrics_rf['R2']:.4f}\n")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Metrics saved to: {metrics_path}")


def save_feature_importance(results_dir, model, feature_cols):
    """
    Save feature importance bar chart to results/feature_importance.png
    """
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_. Skip plotting.")
        return

    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\nFeature importance:")
    print(fi_df)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=fi_df, x="importance", y="feature")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()

    fig_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Feature importance plot saved to: {fig_path}")


def main():
    data_path, results_dir = get_paths()

    # 1. Load data
    df = load_data(data_path)

    # 2. Prepare features
    X, y, feature_cols = prepare_features(df)

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 4. Baseline: Linear Regression
    print("\n=== Training Linear Regression ===")
    lr_model, y_pred_lr, metrics_lr = train_linear_regression(
        X_train, y_train, X_test, y_test
    )
    print("Linear Regression metrics:", metrics_lr)

    # 5. Random Forest with GridSearch
    print("\n=== Training Random Forest ===")
    rf_model, y_pred_rf, metrics_rf = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    print("Random Forest metrics:", metrics_rf)

    # 6. Save metrics and plots
    save_metrics(results_dir, metrics_lr, metrics_rf)
    save_feature_importance(results_dir, rf_model, feature_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
