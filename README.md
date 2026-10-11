# DeliveryETA — Predictive Modeling of Delivery Times

## 1. Project Overview

**DeliveryETA** is a Machine Learning project that predicts the total time required to deliver an order, from preparation to customer receipt.

The goal is to build a regression model that estimates delivery duration (in minutes) based on order characteristics, delivery person attributes, weather conditions, and traffic density.

This solution helps delivery companies improve their Estimated Time of Arrival (ETA), optimize delivery scheduling, and better understand the factors that influence delivery times.

### Business Objectives

- Predict delivery time in minutes.
- Provide customers with an estimated delivery time.
- Identify the factors that influence delivery duration.
- Facilitate route planning and delivery person assignment.
- Compare and optimize multiple regression models.
- Provide an interactive web application for real-time predictions.

---

## 2. Technologies

| Technology | Purpose |
|---|---|
| Python | Core development language |
| Pandas & NumPy | Data cleaning, manipulation, and feature engineering |
| Matplotlib & Seaborn | Data visualization and exploratory analysis |
| Scikit-learn | Preprocessing, modeling, and hyperparameter tuning |
| HistGradientBoostingRegressor | Final selected regression model |
| Pipeline & ColumnTransformer | End-to-end preprocessing + model pipeline |
| RandomizedSearchCV | Hyperparameter optimization |
| Joblib | Model serialization and loading |
| Streamlit | Interactive web application |

---

## 3. Dataset

The project uses the file `dataset_delivery_time.csv`, which contains a historical record of deliveries with information about orders, delivery persons, delivery conditions, and actual delivery times.

### Variable Categories

- **Delivery person:** age, ratings, experience.
- **Order:** type of order, date and time of order and pickup.
- **Delivery conditions:** weather, traffic density, vehicle type and condition, festival indicator.
- **Location:** GPS coordinates (latitude/longitude) of the restaurant and the customer.
- **Target variable:** `Time_taken(min)` — actual delivery time in minutes.

### Data Source

File: `data/raw/dataset_delivery_time.csv`

---

## 4. Project Structure

```text
DeliveryETA/
│
├── data/
│   ├── raw/
│   │   └── dataset_delivery_time.csv          # Raw dataset
│   └── processed/
│       ├── dataset_delivery_time_clean.csv     # Cleaned dataset
│       ├── dataset_delivery_time_model_ready.csv  # Feature-engineered dataset
│       └── model_comparison_results.csv        # Model comparison table
│
├── notebooks/
│   ├── data_cleaning (2).ipynb                 # Data cleaning notebook
│   ├── eda.ipynb                               # Exploratory data analysis
│   └── modeling.ipynb                          # Model training and evaluation
│
├── src/
│   ├── preprocessing/
│   │   └── cleaning.py                         # DataCleaner class
│   ├── features/
│   │   └── engineering.py                      # Feature engineering functions
│   └── modeling/
│       ├── train.py                            # Model comparison pipeline
│       └── tune.py                             # Hyperparameter tuning & final model
│
├── models/
│   ├── model_final.joblib                      # Serialized final pipeline
│   ├── scaler.joblib                           # Extracted StandardScaler
│   ├── metrics.json                            # Final model metrics
│   └── test_predictions.csv                    # Predictions on the test set
│
├── app/
│   ├── streamlit_app.py                        # Streamlit web application
│   └── requirements.txt                        # Python dependencies
│
├── api/                                        # Reserved for future API development
│
└── README.md
```

---

## 5. Architecture & Data Flow

The project follows a sequential pipeline architecture. Each step produces an output that is consumed by the next.

```text
┌───────────────────────────┐
│  data/raw/                │
│  dataset_delivery_time.csv│
└─────────┬─────────────────┘
          │
          ▼
┌───────────────────────────────────┐
│  src/preprocessing/cleaning.py    │
│  DataCleaner class                │
│  • Text noise removal            │
│  • Type conversion               │
│  • Missing value imputation       │
│  • Outlier removal (IQR)         │
│  • Duplicate removal             │
└─────────┬─────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────┐
│  data/processed/                          │
│  dataset_delivery_time_clean.csv          │
└─────────┬─────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────┐
│  src/features/engineering.py      │
│  Feature engineering functions    │
│  • Distance_km (Haversine)       │
│  • Preparation_time_min          │
│  • Order_hour, Day_of_week       │
│  • Is_weekend                    │
│  • Column selection              │
└─────────┬─────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────┐
│  data/processed/                          │
│  dataset_delivery_time_model_ready.csv    │
└─────────┬─────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────┐     ┌──────────────────────────────┐
│  src/modeling/train.py            │     │  src/modeling/tune.py        │
│  Compare 8 regression models     │     │  RandomizedSearchCV on       │
│  Output: comparison table        │     │  HistGradientBoosting        │
└─────────┬─────────────────────────┘     │  Output: final pipeline      │
          │                               └──────────┬───────────────────┘
          ▼                                          ▼
┌─────────────────────────────┐     ┌──────────────────────────────┐
│  data/processed/            │     │  models/                     │
│  model_comparison_results.csv│    │  model_final.joblib          │
└─────────────────────────────┘     │  scaler.joblib               │
                                    │  metrics.json                │
                                    │  test_predictions.csv        │
                                    └──────────┬───────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────────────┐
                                    │  app/streamlit_app.py        │
                                    │  Loads model_final.joblib,   │
                                    │  metrics.json, and           │
                                    │  test_predictions.csv        │
                                    │  for interactive predictions │
                                    └──────────────────────────────┘
```

---

## 6. Component Details

### 6.1 Data Cleaning — `src/preprocessing/cleaning.py`

The `DataCleaner` class implements a full cleaning pipeline via its `run()` method:

1. **Load** the raw CSV file.
2. **Clean text noise** — strip whitespace from categorical columns, remove the `"conditions"` prefix from weather values, extract numeric values from `Time_taken(min)`.
3. **Convert types** — cast age, ratings, and multiple deliveries to numeric; parse dates and times.
4. **Handle missing values** — impute numeric columns with the median, categorical and time columns with the mode.
5. **Handle outliers** — correct negative GPS coordinates (absolute value), remove rows with zero coordinates, remove `Time_taken(min)` outliers using the IQR method.
6. **Remove duplicates** — drop duplicates (excluding the `ID` column).

Output: `data/processed/dataset_delivery_time_clean.csv`

### 6.2 Feature Engineering — `src/features/engineering.py`

Creates derived features and selects model-ready columns:

| Feature | Calculation |
|---|---|
| `Distance_km` | Haversine distance between restaurant and customer GPS coordinates |
| `Preparation_time_min` | Time difference between order time and pickup time (clipped to 0–60 min) |
| `Order_hour` | Hour extracted from order time |
| `Day_of_week` | Day of the week (0 = Monday, 6 = Sunday) |
| `Is_weekend` | Binary indicator (1 if Saturday or Sunday) |

**Dropped columns:** `ID`, `Delivery_person_ID`, all 4 GPS coordinates, `Order_Date`, `Time_Orderd`, `Time_Order_picked`, `Delivery_person_Ratings`.

Output: `data/processed/dataset_delivery_time_model_ready.csv`

### 6.3 Model Training — `src/modeling/train.py`

Compares **8 regression models** on the same 80/20 train/test split:

| Model | Description |
|---|---|
| LinearRegression | Baseline linear model |
| Ridge | L2-regularized linear regression |
| Lasso | L1-regularized linear regression |
| DecisionTreeRegressor | Single decision tree |
| RandomForestRegressor | Ensemble of 200 trees |
| ExtraTreesRegressor | Extremely randomized trees (200 estimators) |
| GradientBoostingRegressor | Gradient boosting |
| HistGradientBoostingRegressor | Histogram-based gradient boosting |

Each model is wrapped in a `Pipeline` with a `ColumnTransformer` preprocessor:
- **Numeric features** → `StandardScaler`
- **Ordinal feature** (`Road_traffic_density`) → `OrdinalEncoder` with order `[Low, Medium, High, Jam]`
- **Nominal features** → `OneHotEncoder` (binary columns dropped)

Output: `data/processed/model_comparison_results.csv`

### 6.4 Hyperparameter Tuning — `src/modeling/tune.py`

Optimizes the **HistGradientBoostingRegressor** using `RandomizedSearchCV`:

- **Search space:** `learning_rate`, `max_iter`, `max_leaf_nodes`, `max_depth`, `min_samples_leaf`, `l2_regularization`
- **Search:** 20 iterations, 3-fold cross-validation, scored on `neg_root_mean_squared_error`
- **Subsampling:** If the training set exceeds 8,000 rows, a random sample is used for the search to speed up tuning.

After finding the best hyperparameters, the final model is trained on the full train split and evaluated on the test split. The outputs are saved to the `models/` directory.

### 6.5 Streamlit Application — `app/streamlit_app.py`

An interactive web interface that loads the trained model for real-time predictions.

**Input fields (sidebar):**

| Input | Type |
|---|---|
| Distance (km) | Number (0–100) |
| Weather | Select: Sunny, Stormy, Sandstorms, Windy, Cloudy, Fog |
| Traffic | Select: Low, Medium, High, Jam |
| Festival | Select: No, Yes |
| Delivery person age | Number (18–70) |
| Vehicle condition | Slider (0–3) |
| Vehicle type | Select: motorcycle, scooter, electric_scooter, bicycle |
| Multiple deliveries | Number (0–3) |
| City | Select: Urban, Metropolitian, Semi-Urban |
| Preparation time (min) | Number (0–120) |
| Order hour | Slider (0–23) |
| Order date | Date picker |
| Order type | Select: Drinks, Buffet, Meal, Snack |

**Application tabs:**

| Tab | Content |
|---|---|
| Données | List of all features used by the model |
| Métriques | MAE, RMSE, R² test, R² adjusted, R² train, overfitting analysis, best hyperparameters |
| Précision | First 20 test predictions with actual vs. predicted values and error analysis |

---

## 7. Results & Performance

### Model Comparison (Initial Training)

All 8 models evaluated on the same test set, sorted by MAE:

| Model | MAE (min) | RMSE (min) | R² test | R² train |
|---|---:|---:|---:|---:|
| HistGradientBoosting | 3.339 | 4.144 | 0.7966 | 0.8085 |
| RandomForest | 3.393 | 4.260 | 0.7850 | 0.9702 |
| ExtraTrees | 3.475 | 4.403 | 0.7703 | 1.0000 |
| GradientBoosting | 3.869 | 4.865 | 0.7196 | 0.7229 |
| DecisionTree | 4.474 | 5.820 | 0.5988 | 1.0000 |
| LinearRegression | 5.140 | 6.441 | 0.5086 | 0.5155 |
| Ridge | 5.140 | 6.441 | 0.5086 | 0.5155 |
| Lasso | 5.146 | 6.448 | 0.5075 | 0.5147 |

### Final Optimized Model (HistGradientBoosting)

After hyperparameter tuning with `RandomizedSearchCV`:

| Metric | Value |
|---|---:|
| MAE | 3.336 min |
| RMSE | 4.142 min |
| R² test | 0.7968 |
| R² adjusted (test) | 0.7961 |
| R² train | 0.8076 |
| Number of features | 27 |
| Training samples | 33,364 |
| Test samples | 8,342 |

**Best hyperparameters:**

| Parameter | Value |
|---|---:|
| learning_rate | 0.05 |
| max_iter | 100 |
| max_depth | 10 |
| max_leaf_nodes | 50 |
| min_samples_leaf | 10 |
| l2_regularization | 0.1 |

### Interpretation

- The **MAE of 3.34 minutes** means predictions deviate on average by about 3.3 minutes from the actual delivery time.
- The **R² of ~0.80** indicates the model explains approximately 80% of the variance in delivery times.
- The **small gap between R² train (0.808) and R² test (0.797)** confirms the model generalizes well without significant overfitting.

### Saved Artifacts

| File | Description |
|---|---|
| `models/model_final.joblib` | Serialized sklearn Pipeline (preprocessor + model) |
| `models/scaler.joblib` | Extracted StandardScaler from the pipeline |
| `models/metrics.json` | Final model metrics and best hyperparameters |
| `models/test_predictions.csv` | Actual vs. predicted values on the test set |

---

## 8. Installation & Setup

### Prerequisites

- Python 3.10+
- pip
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/abderrazakerrifaouy/DeliveryETA-.git
cd DeliveryETA-
```

### Step 2 — Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r app/requirements.txt
```

### Step 4 — Run the Data Pipeline (Optional)

If you want to regenerate the processed data and retrain the model from scratch:

```bash
# Step 1: Clean the raw data
python -m src.preprocessing.cleaning

# Step 2: Generate features
python -m src.features.engineering

# Step 3: Compare models
python -m src.modeling.train

# Step 4: Tune and save the final model
python -m src.modeling.tune
```

> **Note:** The `models/` directory already contains a pre-trained model. Steps above are only needed if you want to retrain.

### Step 5 — Launch the Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

The application will be available at: **http://localhost:8501**

---

## 9. Limitations & Future Improvements

### Limitations

- Prediction quality depends on the representativeness of the historical data.
- Real-world traffic conditions and unforeseen events can affect delivery times in ways the model cannot capture.
- Test set performance does not guarantee identical results in production.
- Tree-based models (ExtraTrees, DecisionTree) show signs of overfitting (R² train = 1.0), confirming HistGradientBoosting as the better choice.
- The model should be periodically retrained to account for data drift.

### Future Improvements

- Enrich the dataset with real-time traffic data.
- Expose the model via a REST API (FastAPI) in the `api/` directory.
- Add unit and integration tests.
- Set up a CI/CD pipeline.
- Implement model monitoring in production.
- Add experiment tracking (e.g., MLflow).

---

## 10. Author & Context

**Project:** DeliveryETA — Predictive Modeling of Delivery Times  
**Instructor:** Hamid OUFAKIR  
**Brief creation date:** 17/09/2026  
**Planned period:** 21/09/2026 – 25/09/2026  
**Certification:** RNCP — AI Developer  
**Type:** Individual project

**GitHub Repository:** [DeliveryETA-](https://github.com/abderrazakerrifaouy/DeliveryETA-)
