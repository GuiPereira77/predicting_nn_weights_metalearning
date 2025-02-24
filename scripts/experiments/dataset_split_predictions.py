import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Convert JSON to CSV
file_to_run = "scripts/experiments/json_to_csv.py"
with open(file_to_run, "r") as file:
    exec(file.read())

# Load CSV file
df = pd.read_csv("scripts/experiments/model_stats.csv")

# Encode categorical variables
encoder = LabelEncoder()
df["dataset_name"] = encoder.fit_transform(df["dataset_name"])
df["dataset_group"] = encoder.fit_transform(df["dataset_group"])
df["scaler_type"] = encoder.fit_transform(df["scaler_type"])

# Handle missing values
df.fillna(0, inplace=True)

# Create unique group identifier for dataset_name & dataset_group
df["dataset_group_id"] = df["dataset_name"].astype(str) + "_" + df["dataset_group"].astype(str)
df["dataset_group_id"] = LabelEncoder().fit_transform(df["dataset_group_id"])

# Define Features & Target
X = df.drop(columns=["id", "smape", "is_better", "dataset_group_id"])
y = df["is_better"]

# Stratified Group K-Fold for Cross-Validation
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
all_reports = []
feature_importances = []

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=df["dataset_group_id"])):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    all_reports.append(f"Fold {fold+1} Accuracy: {acc:.4f}\n{classification_report(y_test, y_pred, zero_division=1)}")
    
    # Store feature importance for averaging
    feature_importances.append(model.feature_importances_)

# Compute mean accuracy and std deviation
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Compute average feature importance
avg_feature_importance = np.mean(feature_importances, axis=0)
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": avg_feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Save results to log file
log_file = "scripts/experiments/dataset_split_predictions.log"
sys.stdout = open(log_file, "w")

print("Dataframe Info:")
print(df.info())

print(f"\nMean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
print("\nCross-Validation Results:")
for report in all_reports:
    print(report)

print("\nAverage Feature Importance:\n", feature_importance_df.to_string(index=False))

# Reset stdout back to default
sys.stdout.close()
sys.stdout = sys.__stdout__

print("Cross-validation results saved to:", log_file)