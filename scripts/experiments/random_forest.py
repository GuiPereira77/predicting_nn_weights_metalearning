import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import sys

# Load JSON file
with open("scripts/experiments/model_stats.json", "r") as f:
    data = json.load(f)

# Convert JSON into structured DataFrame
records = []
for model_name, details in data.items():
    record = {
        "id": model_name,
        "dataset_name": details["dataset"]["name"],
        "dataset_group": details["dataset"]["group"],
        "input_size": details["model"]["input_size"],
        "horizon": details["model"]["horizon"],
        "num_layers": details["model"]["num_layers"],
        "hidden_size": details["model"]["hidden_size"],
        "max_steps": details["model"]["max_steps"],
        "learning_rate": details["model"]["learning_rate"],
        "batch_size": details["model"]["batch_size"],
        "scaler_type": details["model"]["scaler_type"],
        "total_params": details["model"]["total_params"],
        "smape": details["scores"]["smape"],
        "is_better": details["scores"]["is_better"]
    }

    # Extract weight statistics for each layer
    for layer, weights in details["weights"].items():
        record[f"{layer}_mean"] = weights["mean"]
        record[f"{layer}_median"] = weights["median"]
        record[f"{layer}_std"] = weights["std"]
        record[f"{layer}_max"] = weights["max"]
        record[f"{layer}_min"] = weights["min"]

    records.append(record)

df = pd.DataFrame(records)

# Store the DataFrame in a CSV file
df.to_csv("scripts/experiments/model_stats.csv", index=False)

# Encode categorical variables
encoder = LabelEncoder()
df["dataset_name"] = encoder.fit_transform(df["dataset_name"])
df["dataset_group"] = encoder.fit_transform(df["dataset_group"])
df["scaler_type"] = encoder.fit_transform(df["scaler_type"])

# Handle missing values
df.fillna(0, inplace=True)

# Split data
X = df.drop(columns=["id", "smape", "is_better"])  # Features: model parameters + weight stats
target = "is_better"  # "smape" or "is_better"
y = df[target]  # Target: SMAPE scores or binary classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
# model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Show feature importance
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})

# Save model details, predictions, and feature importance to a log file
# Redirect stdout to a file
log_file = "scripts/experiments/output.log"
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

print(f"\nMean Absolute Error (MAE) of {target.upper()} predictions: {mae:.4f}")
print("\nFeature Importance:\n", feature_importance.sort_values(by="Importance", ascending=False).to_string())

# Reset stdout back to default
sys.stdout.close()
sys.stdout = sys.__stdout__

print("Model info saved to:", log_file)