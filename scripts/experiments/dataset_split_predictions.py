import pandas as pd
import numpy as np
import sys
import time
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

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
X = df.drop(columns=["id", "smape", "is_better", "dataset_name", "dataset_group", "dataset_group_id"])

CLASSIFICATION = False
y = df["is_better"] if CLASSIFICATION else df["smape"]

# Group K-Fold for Cross-Validation
num_groups = df["dataset_group_id"].nunique()
cv = GroupKFold(n_splits=num_groups)

scores = []
all_reports = []
feature_importances = []

# Define model and evaluation metric based on the task
if CLASSIFICATION:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    score_name = "Accuracy"
else:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    score_name = "MAE"

def evaluate_model(y_test, y_pred, classification):
    if classification:
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1)
        return acc, report
    else:
        mae_score = mean_absolute_error(y_test, y_pred)
        return mae_score, None
    
# Start Time
start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=df["dataset_group_id"])):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Print test dataset_group_id
    test_group_id = df.iloc[test_idx]["dataset_group_id"].unique()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score, report = evaluate_model(y_test, y_pred, CLASSIFICATION)
    scores.append(score)
    all_reports.append(f"Testing {test_group_id} {score_name}: {score:.4f}\n{report}")

    # Store feature importance for averaging
    feature_importances.append(model.feature_importances_)

# Compute mean and std deviation of the scores
mean_score = np.mean(scores)
std_score = np.std(scores)

# Compute average feature importance
avg_feature_importance = np.mean(feature_importances, axis=0)
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Save results to log file
log_file = f"scripts/experiments/predictions_{'classification' if CLASSIFICATION else 'regression'}.txt"
sys.stdout = open(log_file, "w")

print("Dataframe Info:")
print(df.info())

print(f"\nMean {score_name}: {mean_score:.4f}\nStd {score_name}: {std_score:.4f}")
print("\nCross-Validation Results:")
for report in all_reports:
    print(report)

print("\nAverage Feature Importance:\n", feature_importance_df.to_string(index=False))

print(f"\nExecution Time: {time.time() - start_time:.3f} seconds")

# Reset stdout back to default
sys.stdout.close()
sys.stdout = sys.__stdout__

print("Cross-validation results saved to:", log_file)