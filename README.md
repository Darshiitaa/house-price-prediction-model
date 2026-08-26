# 🏠 House Price Prediction Model — End-to-End ML & Web Application

An end-to-end beginner-friendly Machine Learning project for predicting house prices using Python, Scikit-Learn, Pandas, NumPy, Joblib, and Streamlit.

---

## 📌 Project Overview

This project demonstrates a complete Machine Learning workflow from raw real estate data to an interactive web interface:

**Dataset → Data Understanding → Data Cleaning → Exploratory Data Analysis (EDA) → Preprocessing Pipeline → Train/Test Split → Model Training & Evaluation → Model Saving → Prediction Function → Streamlit Web UI**

---

## 🎯 Problem Statement

Determining the market value of a house based on physical parameters (living space, bedrooms, bathrooms, floors, construction quality, location, etc.) is a classic **supervised regression problem**. This application helps home buyers, sellers, and real estate enthusiasts estimate property values accurately based on historical sales data.

---

## 📊 Dataset Information

- **Records:** 6,579 real house sales
- **Features:** 20 property attributes
- **Target Variable:** `price` (USD $)

### Key Features Used:
- `sqft_living`: Built-up living area in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `floors`: Number of floors/stories
- `sqft_lot`: Total lot area in square feet
- `condition`: Overall condition rating (1 to 5)
- `grade`: Construction quality grade (1 to 13)
- `yr_built`: Year property was constructed
- `waterfront`: Waterfront view indicator (0 = No, 1 = Yes)
- `view`: View quality score (0 to 4)
- `sqft_above`: Above-ground square feet
- `sqft_basement`: Basement square feet
- `zipcode`: Postal code / neighborhood location
- `lat`, `long`: Geographical coordinates

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn (Linear Regression, Decision Tree, Random Forest, ColumnTransformer, Pipeline, Imputer, StandardScaler, OneHotEncoder)
- **Model Serialization:** Joblib
- **Web Interface:** Streamlit
- **Notebook Environment:** Jupyter

---

## 📁 Project Structure

```text
house-price-prediction-model/
│
├── data/
│   └── house_data.csv             # Raw real estate sales dataset
│
├── notebooks/
│   └── house_price_analysis.ipynb # EDA & data visualization notebook
│
├── src/
│   ├── data_preprocessing.py      # Data profiling, pipeline builder & train/test splitter
│   ├── train_model.py             # Model training, comparison & saving script
│   └── predict.py                 # Prediction function & input validation module
│
├── models/
│   └── house_price_model.pkl      # Saved Scikit-Learn trained Pipeline (Joblib binary)
│
├── app.py                         # Interactive Streamlit Web Application
├── requirements.txt               # Python package dependencies
├── README.md                      # Comprehensive documentation & educational guide
└── .gitignore                     # Files ignored by Git
```

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/house-price-prediction-model.git
cd house-price-prediction-model
```

### 2. Set Up Virtual Environment

**Using Miniconda / Conda (Recommended):**
```bash
# Create a conda environment with Python 3.10
conda create -n house-price python=3.10 -y

# Activate the environment
conda activate house-price
```

**Using Standard Python venv (Linux / macOS):**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Using Standard Python venv (Windows):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Running the Project

### Train the Models & Save the Pipeline
```bash
python src/train_model.py
```

### Test the Prediction Function
```bash
python src/predict.py
```

### Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📈 Model Performance Comparison

Three candidate regression algorithms were trained on 80% split data (5,263 samples) and evaluated on 20% holdout test data (1,316 samples):

| Model | MAE ($) | RMSE ($) | R² Score | Performance |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** ⭐ | **$75,582.28** | **$144,538.20** | **0.8516** | **Best Model** |
| **Linear Regression** | $99,934.98 | $172,767.92 | 0.7880 | Baseline |
| **Decision Tree Regressor** | $100,809.36 | $183,942.28 | 0.7597 | Decision Rules |

### Selected Model:
**Random Forest Regressor** achieved the highest accuracy (**85.16% variance explained**) and lowest Mean Absolute Error (**$75,582**). It was saved as `models/house_price_model.pkl`.

---

## 🔮 Example Prediction

Input Parameters:
- Living Area: 2,400 sq ft
- Bedrooms: 4
- Bathrooms: 2.5
- Floors: 2.0
- Year Built: 2005
- Construction Grade: 8
- Zipcode: 98103

Output:
```text
Estimated House Price: $819,384.88
```

---

## 🎓 Educational ML Concepts Explained

### 1. What is Regression?
Regression is a type of **Supervised Learning** where the goal is to predict a continuous numerical value (e.g., house price in dollars) based on input attributes.

### 2. Features vs Target
- **Features ($X$):** The input variables describing property characteristics (area, bedrooms, bathrooms, location).
- **Target ($y$):** The target outcome we want to predict (`price`).

### 3. Why Train/Test Split?
We split data (80% train, 20% test) to evaluate how well our model generalizes to **unseen, real-world data**, preventing memorization (overfitting).

### 4. What is Data Leakage & How Pipeline Prevents It?
**Data Leakage** occurs when information from the testing set inadvertently leaks into training (e.g., scaling using the mean/std of the full dataset). Using Scikit-Learn `Pipeline` and `ColumnTransformer` guarantees that fitting (calculating means, scalers, imputers) happens strictly on the training set.

### 5. Why Encode Categorical Variables?
ML algorithms perform linear algebra and calculus operations, requiring numeric inputs. One-Hot Encoding converts categorical variables (like `zipcode` or `waterfront`) into binary columns (0s and 1s).

### 6. Metric Definitions
- **MAE (Mean Absolute Error):** Average magnitude of prediction errors in dollars. Lower is better.
- **RMSE (Root Mean Squared Error):** Square root of average squared errors; penalizes large outliers heavily.
- **R² Score (Coefficient of Determination):** Proportion of variance in target variable explained by model features. Values range from 0 to 1 (1.0 = 100% accuracy).

### 7. Why Random Forest Outperformed Decision Tree & Linear Regression?
- Linear Regression assumes strict straight-line relationships.
- A single Decision Tree can easily overfit or underfit noisy patterns.
- **Random Forest** builds an ensemble of hundreds of decision trees using random subsets of data and features, averaging predictions to reduce variance and capture complex, non-linear interactions.

---

## 📌 Limitations & Future Improvements

- **Geographic Coverage:** Dataset covers King County real estate; extending to global/national datasets would increase generalizability.
- **Advanced Ensembles:** XGBoost or LightGBM could be added to boost accuracy further.
- **Deployment:** Containerize using Docker and deploy to Streamlit Community Cloud or AWS.

---

## 📄 License
This project is open-source and available under the MIT License.
