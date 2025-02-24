import json
import pandas as pd

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