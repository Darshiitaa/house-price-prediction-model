"""
Model Training and Evaluation Module for House Price Prediction.

Trains 3 regression models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Evaluates models using MAE, RMSE, and R² Score, selects the best model,
and saves the complete trained Pipeline to models/house_price_model.pkl.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from data_preprocessing import (
    load_data,
    inspect_data,
    prepare_features_and_target,
    get_feature_groups,
    build_preprocessor,
    split_dataset
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate MAE, RMSE, and R2 score for a model on test data."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    }


def train_and_evaluate_all():
    """Main execution workflow for training, evaluating, and saving house price model."""
    print("=" * 60)
    print("      HOUSE PRICE PREDICTION - MODEL TRAINING PIPELINE      ")
    print("=" * 60)

    # 1. Load Data
    data_path = os.path.join("data", "house_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")

    print(f"\n1. Loading dataset from '{data_path}'...")
    df = load_data(data_path)

    # 2. Data Understanding & Profiling
    info = inspect_data(df)
    print(f"\nDataset Statistics:")
    print(f"  - Rows: {info['num_rows']}")
    print(f"  - Columns: {info['num_cols']}")
    print(f"  - Duplicate Rows: {info['duplicates']}")
    print(f"  - Missing Values Total: {sum(info['missing_values'].values())}")
    print(f"  - Target Variable: {info['target_col']}")

    # 3. Feature & Target Separation
    X, y = prepare_features_and_target(df)
    num_cols, cat_cols = get_feature_groups(X)

    print(f"\nFeature Summary:")
    print(f"  - Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"  - Categorical Features ({len(cat_cols)}): {cat_cols}")

    # 4. Reproducible Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain/Test Split:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples:  {X_test.shape[0]}")

    # 5. Build Preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 6. Define Regression Models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=12),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # 7. Train & Evaluate Models
    results = []
    trained_pipelines = {}

    print("\nTraining and evaluating candidate regression models...")
    print("-" * 60)

    for name, regressor in candidate_models.items():
        # Create full pipeline: preprocessing + regressor
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

        # Train pipeline on training data ONLY (prevents data leakage)
        full_pipeline.fit(X_train, y_train)

        # Evaluate on testing data
        metrics = evaluate_model(full_pipeline, X_test, y_test)
        metrics["Model"] = name

        results.append(metrics)
        trained_pipelines[name] = full_pipeline

        print(f"[{name}]")
        print(f"  MAE:      ${metrics['MAE']:,.2f}")
        print(f"  RMSE:     ${metrics['RMSE']:,.2f}")
        print(f"  R² Score: {metrics['R2_Score']:.4f}\n")

    # 8. Model Performance Comparison Table
    results_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2_Score"]]
    results_df = results_df.sort_values(by="R2_Score", ascending=False).reset_index(drop=True)

    print("=" * 60)
    print("              MODEL PERFORMANCE COMPARISON                  ")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # 9. Select Best Model
    best_model_name = results_df.iloc[0]["Model"]
    best_r2 = results_df.iloc[0]["R2_Score"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_model_name} (R² Score: {best_r2:.4f})")
    print("=" * 60)

    # 10. Save Model Pipeline using Joblib
    os.makedirs("models", exist_ok=True)
    model_output_path = os.path.join("models", "house_price_model.pkl")

    joblib.dump(best_pipeline, model_output_path)
    print(f"\nTrained model pipeline successfully saved to '{model_output_path}'!")
    return results_df, best_model_name


if __name__ == "__main__":
    train_and_evaluate_all()
