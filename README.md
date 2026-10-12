# ImpactSense - Earthquake Impact Prediction System

ImpactSense is a Machine Learning-based earthquake impact prediction system that analyzes historical earthquake data from **1995–2023** and predicts earthquake magnitude and severity.

The project combines data preprocessing, exploratory data analysis, machine learning, hyperparameter tuning, explainable AI, geospatial visualization, and an interactive Streamlit dashboard.

## Project Features

- Exploratory Data Analysis (EDA)
- Earthquake magnitude and depth analysis
- Correlation analysis
- Interactive global earthquake map using Folium
- Missing value handling and preprocessing
- Numerical feature scaling
- One-Hot Encoding for categorical features
- Machine Learning model comparison
- Cross-validation
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- SHAP-based explainability
- Earthquake magnitude prediction
- Severity classification
- Interactive Streamlit dashboard

## Machine Learning Models

The following regression models were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- HistGradientBoosting Regressor

Classification models were also evaluated for magnitude category prediction:

- Logistic Regression
- One-vs-Rest Logistic Regression
- Logistic Regression with Cross-Validation
- Decision Tree Classifier

The Decision Tree Classifier achieved approximately **82% test accuracy**.

## Final Model

After model comparison, cross-validation, and hyperparameter tuning, the optimized **Random Forest Regressor** was selected for deployment.

### Best Random Forest Parameters

- `n_estimators = 150`
- `max_depth = 10`
- `max_features = 1.0`

### Final Model Performance

| Metric | Result |
|---|---:|
| Test MAE | 0.0972 |
| Test R² Score | 0.7918 |
| Train MAE | 0.0489 |
| Train R² Score | 0.9527 |

The trained model is stored as:

```text
random_forest_model.pkl
```

## Input Features

The deployed model uses the following features:

```text
cdi
mmi
alert
tsunami
sig
net
nst
dmin
gap
magType
depth
latitude
longitude
daytime
```

## Severity Classification

After predicting the earthquake magnitude, the result is converted into a severity category.

| Magnitude | Severity |
|---|---|
| Less than 6 | Moderate |
| 6 to < 7 | Major |
| 7 to < 8 | Extreme |
| 8 or above | Catastrophic |

## Explainable AI

The project includes model interpretation techniques to understand the contribution of different earthquake parameters.

### Feature Importance

Feature importance was analyzed for tree-based machine learning models.

### SHAP

SHAP (SHapley Additive exPlanations) was used to understand how individual features influence model predictions.

SHAP analysis was performed for:

- Random Forest
- Gradient Boosting
- XGBoost

## Streamlit Dashboard

The Streamlit application provides four main sections:

### 1. Overview & Map

- Total earthquake events
- Average magnitude
- Maximum magnitude
- Catastrophic earthquake count
- Tsunami-triggered earthquake count
- Severity breakdown
- Interactive global earthquake map

### 2. Data Explorer

- Search earthquakes by location
- View filtered earthquake records
- Summary statistics
- Missing value analysis

### 3. Analytics

- Interactive scatter plots
- Distribution analysis
- Severity-based box plots
- Correlation heatmap
- Severity analysis over time

### 4. Predict Severity

Users can enter seismic and geographical parameters to obtain:

- Predicted earthquake magnitude
- Predicted severity category
- Magnitude scale visualization
- Event location preview
- Historical earthquake comparison

## Project Structure

```text
ImpactSense/
│
├── app.py
├── ui.py
├── util.py
├── earthquake_1995-2023.csv
├── random_forest_model.pkl
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- GeoPandas
- Folium
- Streamlit
- Streamlit-Folium
- Joblib
- Google Colab

## Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap geopandas folium streamlit streamlit-folium joblib scipy
```

## How to Run the Project

Make sure the following files are present in the same project directory:

```text
ui.py
util.py
random_forest_model.pkl
earthquake_1995-2023.csv
```

Open the terminal inside the project folder and run:

```bash
python -m streamlit run ui.py
```

Alternatively:

```bash
streamlit run ui.py
```

The application will normally open at:

```text
http://localhost:8501
```

To run the simpler version of the dashboard:

```bash
python -m streamlit run app.py
```

## Project Workflow

```text
Earthquake Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Data Preprocessing
        |
        v
Feature Transformation
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
Cross-Validation
        |
        v
Hyperparameter Tuning
        |
        v
Random Forest Regressor
        |
        v
Magnitude Prediction
        |
        v
Severity Classification
        |
        v
Streamlit Dashboard
```

## Conclusion

ImpactSense demonstrates an end-to-end Machine Learning workflow for earthquake magnitude and severity analysis.

Multiple Machine Learning models were trained and evaluated using MAE, MSE, R² Score, cross-validation, and hyperparameter tuning. The optimized Random Forest Regressor was selected for deployment and integrated with an interactive Streamlit dashboard.

The system also incorporates geospatial visualization and explainable AI techniques to provide better understanding of earthquake data and model predictions.

## Disclaimer

This project is developed for **educational and research purposes only**. It is not an official earthquake forecasting or emergency warning system.
