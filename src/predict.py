"""
Prediction Module for House Price Prediction Model.

Loads the saved joblib Pipeline (preprocessor + trained model)
and exposes a reusable function `predict_house_price()` that validates
raw input features and returns the predicted house price.
"""

import os
import joblib
import pandas as pd


def load_trained_model(model_path: str = None):
    """Load saved model pipeline from disk."""
    if model_path is None:
        model_path = os.path.join("models", "house_price_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model pipeline not found at '{model_path}'. "
            f"Please run 'python src/train_model.py' first."
        )

    model_pipeline = joblib.load(model_path)
    return model_pipeline


def validate_input_data(house_data: dict) -> pd.DataFrame:
    """
    Validate input dictionary and convert it to a single-row DataFrame.
    Fills missing optional features with reasonable defaults if omitted.
    """
    required_fields = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'yr_built']

    for field in required_fields:
        if field not in house_data:
            raise ValueError(f"Missing required input feature: '{field}'")

        if house_data[field] is None or house_data[field] < 0:
            raise ValueError(f"Invalid value for '{field}': must be non-negative.")

    # Create DataFrame from input dictionary
    df_input = pd.DataFrame([house_data])
    return df_input


def predict_house_price(house_info: dict, model_path: str = None) -> float:
    """
    Accepts a dictionary containing house parameters, validates input,
    applies model pipeline, and returns estimated price.

    Example input:
    {
        'bedrooms': 3,
        'bathrooms': 2.0,
        'sqft_living': 1800,
        'sqft_lot': 5000,
        'floors': 1.0,
        'waterfront': 0,
        'view': 0,
        'condition': 3,
        'grade': 7,
        'sqft_above': 1800,
        'sqft_basement': 0,
        'yr_built': 1995,
        'yr_renovated': 0,
        'zipcode': 98103,
        'lat': 47.65,
        'long': -122.35,
        'sqft_living15': 1800,
        'sqft_lot15': 5000
    }
    """
    model_pipeline = load_trained_model(model_path)
    df_input = validate_input_data(house_info)

    # Predict using saved pipeline
    prediction = model_pipeline.predict(df_input)[0]

    # Prices should not be negative
    return max(0.0, float(prediction))


if __name__ == "__main__":
    # Test example input
    sample_house = {
        'bedrooms': 4,
        'bathrooms': 2.5,
        'sqft_living': 2400,
        'sqft_lot': 6000,
        'floors': 2.0,
        'waterfront': 0,
        'view': 0,
        'condition': 4,
        'grade': 8,
        'sqft_above': 2400,
        'sqft_basement': 0,
        'yr_built': 2005,
        'yr_renovated': 0,
        'zipcode': 98103,
        'lat': 47.66,
        'long': -122.34,
        'sqft_living15': 2200,
        'sqft_lot15': 5500
    }

    print("Testing House Price Prediction Function...")
    print("Sample Input Parameters:")
    for k, v in sample_house.items():
        print(f"  {k}: {v}")

    predicted_price = predict_house_price(sample_house)
    print(f"\nEstimated House Price: ${predicted_price:,.2f}")
