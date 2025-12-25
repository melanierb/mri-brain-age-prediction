import argparse
import json
import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from sklearn import pipeline, preprocessing, impute
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train brain age regressor (notebook-faithful).")
    p.add_argument("--data-dir", type=str, default="data", help="Folder with X_train.csv, y_train.csv, X_test.csv")
    p.add_argument("--out-dir", type=str, default="outputs", help="Where to write submission/model/metrics")

    return p.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Loads CSVs as numpy arrays.
    """
    X_train = pd.read_csv(data_dir / "X_train.csv", index_col=0)
    y_train = pd.read_csv(data_dir / "y_train.csv", index_col=0)
    y_train = y_train['y'].values  # convert to 1D array
    X_test = pd.read_csv(data_dir / "X_test.csv", index_col=0)

    return X_train, y_train, X_test


def remove_outliers(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Removes outliers from TRAIN only using IsolationForest.
    """

    outlier_model = pipeline.make_pipeline(
        SimpleImputer(strategy="median"),
        preprocessing.RobustScaler(),
        IsolationForest(contamination=0.1, random_state=42),
    )

    pred = outlier_model.fit_predict(X_train)  # +1 inliers, -1 outliers
    keep = pred > 0

    X_train_f = X_train[keep]
    y_train_f = y_train[keep]

    removed = int((~keep).sum())
    print(f"[outliers] removed {removed}/{len(keep)} ({removed/len(keep):.1%}) with contamination=0.1")

    return X_train_f, y_train_f


def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pipeline.make_pipeline(
        impute.SimpleImputer(strategy="median"),
        preprocessing.StandardScaler(),
    )
    X_train_f = model.fit_transform(X_train)
    X_test_f = model.transform(X_test)

    # transorm back to df for selection logic
    X_train_df = pd.DataFrame(X_train_f, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_f, columns=X_test.columns, index=X_test.index)

    return X_train_df, X_test_df


def select_features(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # KBest scores
    kbest = SelectKBest(score_func=f_regression, k="all")
    kbest.fit(X_train, y_train)

    feature_scores_kbest = pd.DataFrame({
        "feature": X_train.columns,
        "score": np.nan_to_num(kbest.scores_, nan=0.0, posinf=0.0, neginf=0.0),
    })

    # RF importances
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    feature_scores_rf = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_,
    })

    # Normalize and combine
    scaler = MinMaxScaler()
    feature_scores_kbest["normalized_score"] = scaler.fit_transform(feature_scores_kbest[["score"]])
    feature_scores_rf["normalized_importance"] = scaler.fit_transform(feature_scores_rf[["importance"]])

    comparison = feature_scores_kbest.merge(
        feature_scores_rf[["feature", "importance", "normalized_importance"]],
        on="feature",
        how="inner",
    )
    comparison["combined_score"] = (comparison["normalized_score"] + comparison["normalized_importance"]) / 2
    comparison = comparison.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # select top 150 features
    k = 150
    top_k_features = comparison.head(k)["feature"].tolist()

    X_train_selected = X_train[top_k_features].copy()
    X_test_selected = X_test[top_k_features].copy()

    return X_train_selected, X_test_selected, comparison


def create_submission(out_dir: Path, y_pred: np.ndarray, X_test: pd.DataFrame) -> Path:
    out_path = out_dir / "submission.csv"
    id_column = X_test.iloc[:,0].astype(int)
    results = pd.DataFrame({"id": id_column, "y": y_pred})
    results.to_csv(out_path, index=False)

    return out_path


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load data
    X_train, y_train, X_test = load_data(data_dir)
    print("Loaded data:", X_train.shape, y_train.shape, X_test.shape)
    features_total = X_train.shape[1]

    # --- train + eval
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, 
        y_train,
        test_size=0.2, 
        random_state=42
    )
    
    ## preprocessing
    X_train_split, y_train_split = remove_outliers(X_train_split, y_train_split)
    X_train_split, X_val_split = preprocess(X_train_split, X_val_split)
    X_train_split, X_val_split, _ = select_features(X_train_split, y_train_split, X_val_split)

    ## train model
    model = HistGradientBoostingRegressor(loss='squared_error', random_state=42)
    model.fit(X_train_split, y_train_split)

    ## predict
    val_pred = model.predict(X_val_split)
    metrics = {
        "r2": float(r2_score(y_val_split, val_pred)),
        "mae": float(mean_absolute_error(y_val_split, val_pred)),
        "n_train": int(len(X_train_split)),
        "n_val": int(len(X_val_split)),
        "n_features_total": features_total,
        "n_features_selected": int(X_train_split.shape[1]),
        "model": "HistGradientBoostingRegressor",
        "seed": 42,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metrics:", metrics)


    # --- fit final preprocessing on full train, apply to test, train final model, predict
    ## preprocessing
    X_train, y_train = remove_outliers(X_train, y_train)
    X_train, X_test = preprocess(X_train, X_test)
    X_train, X_test, feature_ranking = select_features(X_train, y_train, X_test)
    feature_ranking.to_csv(out_dir / "feature_ranking.csv", index=False)

    ## train model
    final_model = HistGradientBoostingRegressor(loss='squared_error', random_state=42)
    final_model.fit(X_train, y_train)

    ## predict
    test_pred = final_model.predict(X_test)
    sub_path = create_submission(out_dir, test_pred, X_test)
    print("Saved submission:", sub_path)

    # Save pickle bundle (model + preprocess artifacts + metrics)
    bundle = {"model": final_model, "metrics": metrics}
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Saved model bundle:", out_dir / "model.pkl")


if __name__ == "__main__":
    main()
