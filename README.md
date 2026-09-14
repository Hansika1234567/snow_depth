# Snow Depth Estimation using XGBoost

A machine learning project that estimates snow depth using XGBoost regression and geospatial, terrain, environmental, and temporal features.

---

## Overview

The goal of this project is to estimate snow depth from terrain, vegetation, and temporal features using XGBoost regression.

The workflow includes:
- Data loading and inspection
- Preprocessing and missing value handling
- Date feature extraction
- Feature engineering
- Model training and evaluation
- Feature importance analysis
- SHAP-based model explainability

The dataset contains approximately 224,000 samples. After feature engineering, the final feature matrix contains 40 features.

---

## Dataset

The original dataset file is:

```
final_dataset_without_modis_with_features.csv
```

The notebook originally loaded it from Google Colab. The data file is **not included** in this repository due to its size. If you want to reproduce the experiment, obtain the dataset separately and place it at:

```
data/final_dataset_without_modis_with_features.csv
```

See `data/README.md` for details.

---

## Features and Preprocessing

- The `date` column is converted to datetime format.
- Temporal features extracted: `year`, `month`, `day`, `dayofyear`.
- `Station_ID` and `date` are dropped after feature extraction.
- Numeric missing values are filled using median imputation (per column).

---

## Feature Engineering

Three interaction features are created from terrain and vegetation variables:

| Feature | Formula |
|---|---|
| Elevation_Slope | elevation × slope |
| Tree_Elevation | Percent_Tree_Cover × elevation |
| Terrain_Ruggedness | tri × slope |

The final feature matrix contains **40 features**.

---

## Model

XGBoost regression is used with the following configuration:

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)
```

---

## Results

Evaluation method: 80/20 random train-test split (`random_state=42`).

| Metric | Score |
|---|---|
| MAE | 60.02 |
| RMSE | 83.87 | 
| R² | 0.9711 |

These values reflect model performance on the held-out 20% test set from the random split experiment.

---

## Explainability

- XGBoost built-in feature importances are calculated and the top 15 features are visualized.
- SHAP (SHapley Additive exPlanations) is used to explain individual predictions.
- SHAP analysis is generated using a sample of 100 test observations.

---

## Visual Results

Feature importance (top 15):

![Feature Importance](results/feature_importance.png)

Actual vs Predicted:

![Actual vs Predicted](results/actual_vs_predicted.png)

---

## Project Structure

```
snow-depth-estimation/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── snow_depth_xgboost.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── metrics.txt
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
└── data/
    └── README.md
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Make sure the dataset is placed at `data/final_dataset_without_modis_with_features.csv`, then run:

```bash
python src/train.py
```

This will preprocess the data, train the model, print evaluation metrics, and save the plots to `results/`.

Alternatively, open and run the notebook:

```
notebooks/snow_depth_xgboost.ipynb
```

---

## Limitations

- Results depend on the dataset and the train-test split methodology used.
- The reported R² of 0.9711 comes from a random 80/20 split and may not reflect performance on unseen time periods.
- The original dataset is not included in this repository.
- This is a machine learning research/academic project and is not intended for production deployment.

---

## Future Improvements

- Evaluate using time-based (chronological) train-test splits.
- Perform systematic hyperparameter tuning.
- Compare performance against other regression models.
- Improve experiment tracking and reproducibility.
- Investigate model generalization across different seasons or geographic regions.

---

## Author
Hansika Jatoth

