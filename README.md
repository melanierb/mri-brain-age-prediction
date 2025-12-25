# 🧠 Predicting Brain Age from MRI-Derived Features

This project addresses the task of predicting a person’s **chronological age** from **MRI-derived brain features**.  
The dataset consists of high-dimensional anatomical measurements extracted from structural MRI scans.

---

## Dataset

- ~800 anatomical features derived from MRI brain scans  
- Tabular (CSV) format, pre-extracted using neuroimaging pipelines  
- Common challenges:
  - Missing values  
  - Outliers and extreme measurements  
  - Irrelevant or redundant features  

> ⚠️ The raw imaging data is **not included** in this repository.

---

## Approach

The pipeline closely mirrors an **exploratory notebook workflow** and is implemented as a **reproducible training script**.  
All preprocessing steps that affect training are fitted on the training data only and reapplied consistently.

### 1. Missing Value Handling

Missing values are imputed using **median imputation**, which is robust to skewed and heavy-tailed feature distributions commonly observed in biomedical data.

---

### 2. Outlier Handling

Outliers are detected using **IsolationForest**, applied **only to the training data**.

- Designed to identify atypical feature patterns  
- Helps reduce the influence of extraction artifacts or corrupted measurements  
- Validation and test samples are *not* filtered to avoid information leakage  

---

### 3. Feature Selection

To reduce dimensionality and remove irrelevant features, a **combined feature ranking** strategy is used:

- **Univariate statistical testing** using `SelectKBest` with `f_regression`  
- **Tree-based feature importance** using a `RandomForestRegressor`  
- Both scores are min-max normalized and averaged into a **combined score**  
- The **top 150 features** are retained  

This approach balances:
- individual feature–target correlation  
- multivariate, non-linear importance captured by tree models  

It improves robustness and mitigates overfitting in high-dimensional, noisy settings.

---

### 4. Regression Model

The final model is a **Histogram-based Gradient Boosting Regressor** (`HistGradientBoostingRegressor`), chosen for:

- Strong performance on tabular data  
- Robustness to non-linear relationships  
- Efficient training on medium-sized datasets  

---

## Evaluation

- **Metric:** Coefficient of determination (R²)  
- A **train/validation split** is used for local evaluation  
- Final evaluation metrics are written to:
