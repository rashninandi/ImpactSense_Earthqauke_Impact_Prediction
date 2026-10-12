# ImpactSense - Earthquake Impact Prediction System

## 1. Problem Statement

The objective of this project is to build a machine learning-based predictive model that estimates the impact of earthquakes in terms of magnitude, damage level, or risk zone classification. The system will utilize geophysical and environmental data like latitude, longitude, depth, seismic wave features, and geological parameters to assess the possible damage or categorize the earthquake severity. This can help in early disaster planning, emergency response, and awareness campaigns.

ImpactSense analyzes historical earthquake data from **1995–2023** and applies machine learning techniques to predict earthquake magnitude. The predicted magnitude is further categorized into severity levels to provide an understandable representation of the possible earthquake impact.

---

## 2. Challenges

Developing an earthquake impact prediction system involves several challenges:

- **Complex seismic data:** Earthquake datasets contain numerical, categorical, geographical, and seismic attributes that require different preprocessing techniques.
- **Missing values:** Some earthquake attributes may contain missing observations and require appropriate imputation.
- **Categorical features:** Features such as alert level, network, magnitude type, and daytime need to be transformed into a machine-readable representation.
- **Skewed numerical features:** Certain numerical attributes have skewed distributions and require transformation before model training.
- **Model selection:** Multiple machine learning algorithms need to be compared to identify the most suitable model.
- **Overfitting:** Powerful ensemble models can perform very well on training data but may not generalize equally well to unseen earthquakes.
- **Hyperparameter selection:** Model parameters need to be optimized systematically rather than selected manually.
- **Interpretability:** Understanding which seismic features influence predictions is important for making the system more explainable.
- **Deployment:** The trained machine learning pipeline must be integrated with an interactive interface while maintaining the same preprocessing steps used during training.

---

## 3. Proposed Solution

ImpactSense provides an end-to-end machine learning solution for earthquake impact analysis and magnitude prediction.

The proposed system performs the following operations:

1. Loads historical earthquake data from 1995–2023.
2. Performs Exploratory Data Analysis (EDA) to understand seismic patterns.
3. Preprocesses numerical and categorical features.
4. Handles missing values using appropriate imputation techniques.
5. Applies feature transformations, encoding, and scaling.
6. Trains and compares multiple regression and classification algorithms.
7. Evaluates models using appropriate performance metrics and cross-validation.
8. Optimizes ensemble models using GridSearchCV.
9. Uses feature importance and SHAP for model explainability.
10. Saves the selected machine learning model using Joblib.
11. Integrates the trained model into an interactive Streamlit dashboard.
12. Predicts earthquake magnitude and converts the result into an understandable severity category.

The final optimized **Random Forest Regressor** achieved a test **MAE of approximately 0.0972** and an **R² score of approximately 0.7918** and was selected for deployment.

---

## 4. System Architecture

The overall architecture of ImpactSense is:

```text
                        +---------------------------+
                        |  Earthquake Dataset       |
                        |      (1995-2023)          |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Exploratory Data Analysis |
                        | & Data Visualization      |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |     Data Preprocessing    |
                        |                           |
                        | - Missing Value Handling  |
                        | - Log Transformation      |
                        | - One-Hot Encoding        |
                        | - Feature Scaling         |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Machine Learning Models   |
                        |                           |
                        | - Linear Regression       |
                        | - Decision Tree           |
                        | - Random Forest           |
                        | - Gradient Boosting       |
                        | - XGBoost                 |
                        | - HistGradient Boosting   |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Model Evaluation &        |
                        | Cross-Validation          |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Hyperparameter Tuning     |
                        |      GridSearchCV         |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Tuned Random Forest Model |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Feature Importance & SHAP |
                        |       Explainability      |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |    Saved ML Pipeline      |
                        | random_forest_model.pkl   |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |   Streamlit Application   |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Magnitude + Severity      |
                        |       Prediction          |
                        +---------------------------+
```

---

## 5. Tech Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Machine Learning Techniques

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- HistGradientBoosting
- Cross-Validation
- GridSearchCV

### Explainable AI

- SHAP
- Feature Importance

### Geospatial Visualization

- GeoPandas
- Folium
- Streamlit-Folium

### Deployment and UI

- Streamlit

### Model Serialization

- Joblib

### Development Environment

- Google Colab
- Python

---

## 6. System Workflow

```text
Historical Earthquake Dataset
            |
            v
      Data Loading
            |
            v
Exploratory Data Analysis
            |
            v
     Data Preprocessing
            |
            +--------------------+
            |                    |
            v                    v
     Numerical Data       Categorical Data
            |                    |
      Median Imputation    Frequent-Value
            |               Imputation
      Log Transformation         |
            |              One-Hot Encoding
      Standard Scaling           |
            |                    |
            +---------+----------+
                      |
                      v
              Processed Features
                      |
                      v
              Train-Test Split
                      |
                      v
            Model Development
                      |
                      v
            Model Comparison
                      |
                      v
          5-Fold Cross-Validation
                      |
                      v
              GridSearchCV
                      |
                      v
          Tuned Random Forest
                      |
                      v
          Model Explainability
        Feature Importance + SHAP
                      |
                      v
       random_forest_model.pkl
                      |
                      v
           Streamlit Dashboard
                      |
                      v
       User Enters Seismic Data
                      |
                      v
           Magnitude Prediction
                      |
                      v
          Severity Classification
```

The deployed model uses seismic and geographical attributes including:

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

The predicted magnitude is converted into severity levels:

| Magnitude | Severity |
|---|---|
| Magnitude < 6 | Moderate |
| 6 <= Magnitude < 7 | Major |
| 7 <= Magnitude < 8 | Extreme |
| Magnitude >= 8 | Catastrophic |

---

## 7. Data Analysis

Exploratory Data Analysis was performed to understand the structure, distribution, geographical patterns, and relationships present in the earthquake dataset.

### Magnitude Distribution

A histogram and Kernel Density Estimate (KDE) were used to examine the distribution of earthquake magnitudes.

This helps identify how earthquake events are distributed across different magnitude levels.

### Depth Distribution

Earthquake depth was analyzed using a histogram and KDE curve to understand the distribution of shallow and deep seismic events.

### Tsunami Occurrence

A count plot was used to compare earthquake events based on whether they were associated with a tsunami.

### Alert Level Analysis

Earthquake alert categories were analyzed using a count plot to understand the frequency of different alert levels.

### Correlation Analysis

A correlation heatmap was generated for numerical attributes to identify relationships between earthquake features.

### Geographical Analysis

Latitude and longitude were used to create a GeoDataFrame and visualize earthquake locations across the world.

An interactive **Folium map** was also created to display earthquake events geographically.

### Model Analysis

Multiple regression models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score
- 5-Fold Cross-Validation

The major advanced regression models included:

| Model | Test MAE | Test R² |
|---|---:|---:|
| Random Forest | 0.1510 | 0.5975 |
| Gradient Boosting | 0.1105 | 0.7746 |
| XGBoost | 0.1177 | 0.7467 |
| HistGradientBoosting | 0.1261 | 0.7167 |

Hyperparameter tuning was subsequently performed using GridSearchCV.

The tuned Random Forest produced:

| Metric | Result |
|---|---:|
| Test MAE | 0.0972 |
| Test R² | 0.7918 |
| Training MAE | 0.0489 |
| Training R² | 0.9527 |

Therefore, the tuned Random Forest model was saved as:

```text
random_forest_model.pkl
```

### Explainable AI

Feature importance was used to identify influential features in the tree-based models.

SHAP (SHapley Additive exPlanations) was also applied to analyze the contribution of individual features toward model predictions.

SHAP analysis was performed for:

- Random Forest
- Gradient Boosting
- XGBoost

This provides additional interpretability beyond prediction accuracy alone.

---

## 8. Key Features of the UI

ImpactSense provides an interactive Streamlit dashboard for earthquake analysis and prediction.

### Overview & Map

The overview page provides:

- Total earthquake events
- Average earthquake magnitude
- Maximum earthquake magnitude
- Number of catastrophic events
- Number of tsunami-triggered events
- Severity breakdown
- Interactive global seismic activity map
- Severity-based map markers
- Multiple map styles

### Data Explorer

Users can interactively explore the earthquake dataset through:

- Location-based search
- Configurable number of displayed records
- Magnitude highlighting
- Summary statistics
- Missing-value analysis
- Dataset filtering

### Analytics Dashboard

The analytics section contains:

- Interactive scatter plots
- Distribution histograms
- Severity-wise box plots
- Correlation heatmap
- Severity analysis over time
- Stacked severity visualization

### Earthquake Prediction

Users can enter earthquake parameters such as:

- Latitude
- Longitude
- CDI
- MMI
- Alert level
- Significance
- Seismic network
- Station count
- Minimum station distance
- Azimuthal gap
- Magnitude algorithm
- Depth
- Time of day
- Tsunami indicator

The system then displays:

- Predicted earthquake magnitude
- Predicted severity category
- Earthquake depth
- Tsunami status
- Magnitude scale visualization
- Predicted event location on a map
- Comparison with historical earthquake magnitude data
- Comparison with historical severity distribution

Prediction results are maintained using Streamlit session state so they remain available across normal interface reruns.

---

## 9. Future Scope

ImpactSense can be extended further in several directions:

- **Real-time seismic data integration:** Connect the application with live earthquake monitoring APIs to analyze recent seismic events.

- **Larger datasets:** Train the model using more extensive historical and regional earthquake datasets.

- **Additional geological parameters:** Include soil type, tectonic plate information, fault-line distance, ground acceleration, population density, and infrastructure characteristics.

- **Damage prediction:** Extend the system beyond magnitude prediction to directly estimate infrastructure damage levels.

- **Risk zone classification:** Develop geographical risk classifications such as low-risk, medium-risk, and high-risk earthquake zones.

- **Deep learning models:** Explore neural networks and other deep learning techniques for complex seismic pattern recognition.

- **Improved explainability:** Add interactive SHAP visualizations directly to the Streamlit dashboard.

- **Geospatial risk maps:** Generate regional earthquake vulnerability and risk maps using GIS data.

- **Emergency response support:** Integrate predicted earthquake impact with emergency response recommendations and resource allocation systems.

- **Cloud deployment:** Deploy ImpactSense on a cloud platform so the dashboard can be accessed remotely.

- **Model monitoring:** Monitor prediction performance and retrain the model as new earthquake data becomes available.

---

## 10. Conclusion

ImpactSense provides an end-to-end machine learning framework for analyzing historical seismic data and predicting earthquake magnitude and severity.

The project combines **data analysis, machine learning, ensemble models, cross-validation, hyperparameter tuning, explainable AI, geospatial visualization, and Streamlit deployment** into a single system.

After evaluating multiple machine learning algorithms, the tuned **Random Forest Regressor** was selected for deployment based on its prediction performance. The trained model is integrated into an interactive dashboard that allows users to explore historical earthquakes, analyze seismic characteristics, visualize global earthquake activity, and generate magnitude and severity predictions.

> **Disclaimer:** ImpactSense is developed for educational and research purposes. It should not be treated as an official earthquake forecasting, early-warning, or emergency-response system.
