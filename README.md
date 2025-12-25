# Predicting Brain Age from MRI-Derived Features
This project addresses the task of predicting a person’s chronological age from MRI-derived brain features. The dataset consists of high-dimensional anatomical measurements extracted from structural MRI scans.

The focus of this project is robust preprocessing and reproducible modeling for noisy biomedical tabular data, rather than aggressive model tuning.

## Dataset
- ~800 anatomical features derived from MRI brain scans
- Tabular (CSV) format, pre-extracted using neuroimaging pipelines
- Common challenges:
    - Missing values
    - Outliers / extreme measurements
    - Irrelevant or redundant features
The raw imaging data is not included in this repository.

# Approach
The pipeline closely mirrors an exploratory notebook workflow and is implemented as a reproducible training script.

1. Missing Value Handling
Missing values are imputed using median imputation, which is robust to skewed feature distributions commonly found in biomedical data.

2. Outlier Handling
Outliers are detected using IsolationForest

3. Feature Selection
To reduce dimensionality and remove irrelevant features:

A Random Forest regressor is trained on the preprocessed training data

Feature importance scores are used to select informative features

By default, features with importance above the median are retained
(alternatively, a top-K selection can be used)

This step improves robustness and mitigates overfitting in high-dimensional,
noisy settings.

4. Regression Model

The final model is a Histogram-based Gradient Boosting Regressor
(HistGradientBoostingRegressor), chosen for:

Good performance on tabular data

Robustness to non-linear relationships

Efficient training on medium-sized datasets

Evaluation

Metric: Coefficient of determination (R²)

A train/validation split is used for local evaluation

Final metrics are written to outputs/metrics.json
- R^2

# Getting started
python3 -m venv aml  

source aml/bin/activate

pip install -r requirements.txt

### Run code
python -m train --data-dir data --out-dir outputs