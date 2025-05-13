import pandas as pd
import numpy as np
import time
import sys
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error,
    roc_auc_score, log_loss, f1_score, mean_squared_error, r2_score
)
from scipy.stats import pearsonr, kendalltau, spearmanr
from xgboost import XGBRFClassifier, XGBRFRegressor
import joblib

def load_and_preprocess_data(csv_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(csv_path)
    df["dataset_group_id"] = df["dataset_name"].astype(str) + "_" + df["dataset_group"].astype(str)
    encoder = LabelEncoder()
    df["model_scaler_type"] = encoder.fit_transform(df["model_scaler_type"])
    df = df.astype({col: 'float64' for col in df.select_dtypes(include=['int64']).columns})
    df.fillna(0, inplace=True)
    return df

def define_model(classification):
    """Define the model based on the task type."""
    if classification:
        return XGBRFClassifier(n_estimators=200, random_state=42)
    return XGBRFRegressor(n_estimators=200, random_state=42)

def evaluate_model(y_test, y_pred, y_pred_proba, classification):
    """Evaluates a model's performance based on test data and predictions."""
    if classification:
        score_dict = {
            "acc_score": accuracy_score(y_test, y_pred),
            "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
            "log_loss_score": log_loss(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred)
        }
        return score_dict, classification_report(y_test, y_pred, zero_division=1)
    else:
        score_dict = {
            "mae_score": mean_absolute_error(y_test, y_pred),
            "mse_score": mean_squared_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred),
            "pearson": pearsonr(y_test, y_pred)[0],
            "kendall": kendalltau(y_test, y_pred)[0],
            "spearman": spearmanr(y_test, y_pred)[0]
        }
        return score_dict, None

def cross_validate_model(df, X, y, model, classification):
    """Perform cross-validation and return results."""
    cv = GroupKFold(n_splits=df["dataset_group_id"].nunique())
    scores = {}
    all_reports = []
    feature_importances = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=df["dataset_group_id"])):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_group_id = df.iloc[test_idx]["dataset_group_id"].unique()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)[:, 1] if classification else None

        score_dict, report = evaluate_model(y_test, y_pred, y_pred_proba, classification)
        all_reports.append(f"Testing {test_group_id}:")
        for score_name, score in score_dict.items():
            scores[score_name] = scores.get(score_name, []) + [score]
            all_reports.append(f"- {score_name}: {score:.4f}")
        all_reports.append(report)

        feature_importances.append(model.feature_importances_)

    return scores, all_reports, feature_importances

def save_results(log_file, df, mean_std_df, all_reports, feature_importance_df, execution_time):
    """Save results to a log file."""
    with open(log_file, "w") as log:
        sys.stdout = log
        print("Dataframe Info:")
        print(df.info())
        print("\nMean and Std Scores:")
        print(mean_std_df.to_string(index=False))
        print("\nCross-Validation Results:")
        for report in all_reports:
            print(report)
        print("\nAverage Feature Importance:\n", feature_importance_df.to_string(index=False))
        print(f"\nExecution Time: {execution_time:.3f} seconds")
        sys.stdout = sys.__stdout__

def main():
    classification = False  # Set to True for classification mode

    # Configuration
    output_dir = os.path.join("scripts", "experiments", "output", f"{'classification' if classification else 'regression'}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = "scripts/experiments/model_stats.csv"
    df = load_and_preprocess_data(csv_path)

    shape_columns = df.columns[df.columns.str.contains('shape')]
    columns_to_drop = [
        "id", "model_name", "seed", "dataset_name", "dataset_group",
        "dataset_group_id", "scores_smape", "scores_mse", "scores_mae",
        "scores_r2_score", "scores_sn_smape", "scores_is_better",
        *shape_columns.tolist()
    ]

    # Define target variable
    y = df["scores_is_better"] if classification else df["scores_smape"]

    stages = [
        "start", "step_10", "step_25", "step_50", "step_100",
        "step_200", "step_300", "step_400", "step_500"
    ]

    summary = []

    for i in range(1, len(stages)+1):
        active_stages = stages[:i]
        print(f"Running with stages: {active_stages}")

        # Select only columns related to current stages
        keep_columns = [col for col in df.columns if any(stage in col for stage in active_stages)]
        X = df[keep_columns].copy()
        X = X.drop(columns=columns_to_drop, errors='ignore')

        # Define model
        model = define_model(classification)

        # Run cross-validation
        start_time = time.time()
        scores, all_reports, feature_importances = cross_validate_model(df, X, y, model, classification)
        execution_time = time.time() - start_time

        # Output metrics
        mean_std_df = pd.DataFrame({
            "Metric": scores.keys(),
            "Mean Score": [f"{np.mean(val):.4f}" for val in scores.values()],
            "Std Score": [f"{np.std(val):.4f}" for val in scores.values()]
        })

        mean_std_dict = {}
        for _, row in mean_std_df.iterrows():
            mean_std_dict[f"{row['Metric']}_mean"] = float(row['Mean Score'])
            mean_std_dict[f"{row['Metric']}_std"] = float(row['Std Score'])

        # Add to summary table
        summary.append({
            "stage": active_stages[-1], 
            **mean_std_dict,
            "time": execution_time
        })

        # Calculate feature importance (filter top 10)
        avg_feature_importance = np.mean(feature_importances, axis=0)
        feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
        feature_importance_df = feature_importance_df.head(10) 

        # Save results
        log_file = os.path.join(output_dir, active_stages[-1]+".txt")
        save_results(log_file, X, mean_std_df, all_reports, feature_importance_df, execution_time)

    # Create summary dataframe
    summary_df = pd.DataFrame(summary).set_index("stage")
    summary_df.to_csv(output_dir+"/stagewise_summary.csv")

if __name__ == "__main__":
    main()