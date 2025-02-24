import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, make_scorer
import sys

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

# Prepare data
X = df.drop(columns=["id", "smape", "is_better"])  # Features: model parameters + weight stats
target = "is_better"  # "smape" or "is_better"
y = df[target]  # Target: SMAPE scores or binary classification

# Define the model
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Define scoring metric (MAE in this case)
scoring = make_scorer(mean_absolute_error)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

# Print cross-validation results
print(f"Cross-validation MAE scores: {cv_scores}")
print(f"Mean MAE: {cv_scores.mean():.4f}")
print(f"Standard deviation of MAE: {cv_scores.std():.4f}")

# Fit the model on the entire dataset to get feature importance
model.fit(X, y)

# Show feature importance
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})

# Save model details, predictions, and feature importance to a log file
# Redirect stdout to a file
log_file = "scripts/experiments/cross_validation.txt"
sys.stdout = open(log_file, "w")

print("Dataframe Info:")
print(df.info())

print(f"\nInput Size: {df['input_size'].unique()}")
print(f"Horizon: {df['horizon'].unique()}")
print(f"Num Layers: {df['num_layers'].unique()}")
print(f"Hidden Size: {df['hidden_size'].unique()}")
print(f"Max Steps: {df['max_steps'].unique()}")
print(f"Learning Rate: {df['learning_rate'].unique()}")
print(f"Batch Size: {df['batch_size'].unique()}")
print(f"Scaler Type: {df['scaler_type'].unique()}")

# Print model details
print("\nModel Details:")
print(model)

print("\nModel Parameters:")
print(model.get_params())

print("\nCross-validation MAE scores:", cv_scores)
print(f"Mean MAE: {cv_scores.mean():.4f}")
print(f"Standard deviation of MAE: {cv_scores.std():.4f}")

print("\nFeature Importance:\n", feature_importance.sort_values(by="Importance", ascending=False).to_string())

# Reset stdout back to default
sys.stdout.close()
sys.stdout = sys.__stdout__

print("Model info saved to:", log_file)