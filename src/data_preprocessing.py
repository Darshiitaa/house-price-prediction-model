"""
Data Preprocessing Module for House Price Prediction Model.

This module handles:
1. Loading the dataset
2. Inspecting dataset structure, data types, missing values, and duplicates
3. Separating features (X) and target variable (y)
4. Building Scikit-Learn ColumnTransformer pipeline for preprocessing
5. Train/Test data splitting with reproducibility
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """Load the house dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Perform initial inspection of dataset.
    Returns dictionary with dataset metadata.
    """
    info = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_col": "price"
    }
    return info


def prepare_features_and_target(df: pd.DataFrame):
    """
    Clean dataset and separate features X and target y.
    """
    # Remove exact duplicate rows if present
    df_clean = df.drop_duplicates().copy()

    # Drop non-predictive columns like 'id' and 'date' if present
    cols_to_drop = ['id', 'date']
    existing_drops = [c for c in cols_to_drop if c in df_clean.columns]
    if existing_drops:
        df_clean = df_clean.drop(columns=existing_drops)

    # Separate target variable 'price'
    if 'price' not in df_clean.columns:
        raise KeyError("Target column 'price' not found in dataset.")

    # Drop rows where target variable is missing (if any)
    df_clean = df_clean.dropna(subset=['price'])

    X = df_clean.drop(columns=['price'])
    y = df_clean['price']

    return X, y


def get_feature_groups(X: pd.DataFrame):
    """
    Identify numerical and categorical features.
    """
    # Categorical / discrete location features
    categorical_cols = [c for c in ['zipcode', 'waterfront'] if c in X.columns]
    
    # Numerical features
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Create Scikit-Learn ColumnTransformer pipeline.
    Prevents data leakage by encapsulating imputers and scalers.
    """
    # Numerical Transformer: Median Imputation -> Standard Scaling
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Transformer: Most Frequent Imputation -> One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )

    return preprocessor


def split_dataset(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split feature matrix X and target y into 80% train and 20% test sets.
    Fixed random_state ensures reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
