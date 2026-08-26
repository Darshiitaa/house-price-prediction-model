"""
Streamlit Web Application for House Price Prediction.

Provides a clean, interactive user interface to input house parameters,
predict property value using the trained Scikit-Learn pipeline,
and view model performance metrics.
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_house_price, load_trained_model
from data_preprocessing import load_data, inspect_data


# Page Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .price-card {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .price-title {
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Application Title
    st.markdown('<div class="main-header">🏠 House Price Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An end-to-end Machine Learning web application predicting residential property values.</div>', unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Price Predictor", "Model Performance", "Dataset Overview"])

    # Load dataset for dropdown values and stats
    data_path = os.path.join("data", "house_data.csv")
    if os.path.exists(data_path):
        df_raw = load_data(data_path)
    else:
        df_raw = None

    # PAGE 1: PRICE PREDICTOR
    if page == "Price Predictor":
        st.markdown("### 📋 Enter Property Specifications")
        st.write("Adjust the features below to estimate the expected market value of the house.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📐 Space & Layout")
            sqft_living = st.number_input("Living Area (sq ft)", min_value=300, max_value=15000, value=2000, step=50)
            sqft_lot = st.number_input("Lot Size (sq ft)", min_value=500, max_value=100000, value=5000, step=100)
            bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
            bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.25)
            floors = st.selectbox("Floors / Stories", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=2)

        with col2:
            st.subheader("🏗️ Building Attributes")
            yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=2000)
            condition = st.slider("Property Condition (1=Poor, 5=Excellent)", min_value=1, max_value=5, value=3)
            grade = st.slider("Construction Grade (1=Low, 13=Luxury)", min_value=1, max_value=13, value=7)
            sqft_above = st.number_input("Above-Ground Area (sq ft)", min_value=300, max_value=10000, value=1700, step=50)
            sqft_basement = st.number_input("Basement Area (sq ft)", min_value=0, max_value=5000, value=300, step=50)

        with col3:
            st.subheader("📍 Location & Features")
            waterfront = st.selectbox("Waterfront View", ["No", "Yes"])
            view = st.slider("View Quality Score (0=None, 4=Exceptional)", min_value=0, max_value=4, value=0)
            
            # Zipcode selection
            if df_raw is not None and 'zipcode' in df_raw.columns:
                zipcodes = sorted(df_raw['zipcode'].dropna().unique().astype(int).tolist())
                zipcode = st.selectbox("Zipcode / Location", zipcodes, index=0)
            else:
                zipcode = st.number_input("Zipcode", min_value=98000, max_value=99000, value=98103)

            lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.56, format="%.4f")
            long = st.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.25, format="%.4f")

        # Convert user inputs to dictionary
        input_data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': 1 if waterfront == "Yes" else 0,
            'view': view,
            'condition': condition,
            'grade': grade,
            'sqft_above': sqft_above,
            'sqft_basement': sqft_basement,
            'yr_built': yr_built,
            'yr_renovated': 0,
            'zipcode': float(zipcode),
            'lat': lat,
            'long': long,
            'sqft_living15': sqft_living,
            'sqft_lot15': sqft_lot
        }

        # Predict Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
            try:
                predicted_price = predict_house_price(input_data)
                
                st.markdown(f"""
                <div class="price-card">
                    <div class="price-title">Estimated House Price</div>
                    <div class="price-value">${predicted_price:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ Prediction generated successfully using Random Forest Regressor Pipeline!")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

    # PAGE 2: MODEL PERFORMANCE
    elif page == "Model Performance":
        st.markdown("### 📊 Model Evaluation & Comparison")
        st.write("Comparison of candidate regression algorithms trained on 80% split and tested on 20% holdout test set.")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Selected Model", value="Random Forest")
        with m2:
            st.metric(label="R² Score (Accuracy)", value="85.16%")
        with m3:
            st.metric(label="Mean Absolute Error (MAE)", value="$75,582")

        st.subheader("Model Comparison Metrics Table")
        comparison_data = pd.DataFrame({
            "Model": ["Random Forest Regressor", "Linear Regression", "Decision Tree Regressor"],
            "MAE ($)": [75582.28, 99934.98, 100809.36],
            "RMSE ($)": [144538.20, 172767.92, 183942.28],
            "R² Score": [0.8516, 0.7880, 0.7597]
        })
        st.dataframe(comparison_data.style.highlight_max(subset=["R² Score"], color="#DCFCE7"), use_container_width=True)

        st.subheader("💡 Key Model Insights")
        st.info("""
        - **Random Forest Regressor** achieved the highest **R² Score (0.8516)** and lowest **MAE ($75,582)** because it combines multiple decision trees to capture non-linear relationships.
        - **Linear Regression** performs reasonably well (R²: 0.7880) but assumes linear relationships between features and price.
        - **Decision Tree Regressor** prone to higher variance on unpruned depth.
        """)

    # PAGE 3: DATASET OVERVIEW
    elif page == "Dataset Overview":
        st.markdown("### 📁 Dataset Exploration & Features")
        if df_raw is not None:
            st.write(f"The dataset contains **{df_raw.shape[0]:,} records** and **{df_raw.shape[1]} features** of real estate sales.")

            st.subheader("Sample Raw Data")
            st.dataframe(df_raw.head(10), use_container_width=True)

            st.subheader("Feature Summary Statistics")
            st.dataframe(df_raw.describe().T, use_container_width=True)
        else:
            st.warning("Dataset file `data/house_data.csv` not found.")


if __name__ == "__main__":
    main()
