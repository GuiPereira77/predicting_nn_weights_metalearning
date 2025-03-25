import pandas as pd
import numpy as np
import sys
import time
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, roc_auc_score, log_loss, f1_score, mean_squared_error, r2_score
from xgboost import XGBRFClassifier, XGBRFRegressor

# Convert JSON to CSV
file_to_run = "scripts/experiments/json_to_csv.py"
with open(file_to_run, "r") as file:
    exec(file.read())

# Load CSV file
df = pd.read_csv("scripts/experiments/model_stats.csv")

# Create unique group identifier for dataset_name & dataset_group
df["dataset_group_id"] = df["dataset_name"].astype(str) + "_" + df["dataset_group"].astype(str)

# Encode categorical variables
encoder = LabelEncoder()
df["scaler_type"] = encoder.fit_transform(df["scaler_type"])

# Handle missing values
df.fillna(0, inplace=True)

# Define Features & Target
X = df.drop(columns=["id", "smape", "is_better", "dataset_name", "dataset_group", "dataset_group_id", "seed"])

CLASSIFICATION = False
y = df["is_better"] if CLASSIFICATION else df["smape"]

# Group K-Fold for Cross-Validation
num_groups = df["dataset_group_id"].nunique()
cv = GroupKFold(n_splits=num_groups)
# cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = {}
all_reports = []
feature_importances = []

# Define model and evaluation metric based on the task
if CLASSIFICATION:
    model = XGBRFClassifier(n_estimators=200, random_state=42)
else:
    model = XGBRFRegressor(n_estimators=200, random_state=42)

def evaluate_model(y_test, y_pred, classification):
    if classification:
        score_dict = {
            "acc_score": accuracy_score(y_test, y_pred),
            "roc_auc_score": roc_auc_score(y_test, y_pred),
            "log_loss_score": log_loss(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        return score_dict, classification_report(y_test, y_pred, zero_division=1)
    else:
        score_dict = {
            "mae_score": mean_absolute_error(y_test, y_pred),
            "mse_score": mean_squared_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred)
        }
        return score_dict, None
    
# Start Time
start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=df["dataset_group_id"])):
# for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Print test dataset_group_id
    test_group_id = df.iloc[test_idx]["dataset_group_id"].unique()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score_dict, report = evaluate_model(y_test, y_pred, CLASSIFICATION)
    
    all_reports.append(f"Testing {test_group_id}:")
    for score_name, score in score_dict.items():
        scores[score_name] = scores.get(score_name, []) + [score]
        all_reports.append(f"- {score_name}: {score:.4f}")
    all_reports.append(report)

    # Store feature importance for averaging
    feature_importances.append(model.feature_importances_)

# Compute mean and std deviation of all the scores
mean_std_df = pd.DataFrame({
    "Metric": scores.keys(),
    "Mean Score": [f"{np.mean(scores_list):.4f}" for scores_list in scores.values()],
    "Std Score": [f"{np.std(scores_list):.4f}" for scores_list in scores.values()]
})

# Compute average feature importance
avg_feature_importance = np.mean(feature_importances, axis=0)
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Save results to log file
log_file = f"scripts/experiments/predictions_{'classification' if CLASSIFICATION else 'regression'}.txt"
sys.stdout = open(log_file, "w")

print("Dataframe Info:")
print(df.info())

print("\nMean and Std Scores:")
print(mean_std_df.to_string(index=False))
print("\nCross-Validation Results:")
for report in all_reports:
    print(report)

print("\nAverage Feature Importance:\n", feature_importance_df.to_string(index=False))

print(f"\nExecution Time: {time.time() - start_time:.3f} seconds")

# Reset stdout back to default
sys.stdout.close()
sys.stdout = sys.__stdout__

print("Cross-validation results saved to:", log_file)