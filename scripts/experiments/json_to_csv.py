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
        # "gradient_norm": details["weights"]["gradient_norm"],
        # "model_variance": details["weights"]["model_variance"],
        "seed": details["seed"],
        "smape": details["scores"]["smape"],
        "is_better": details["scores"]["is_better"]
    }

    # Extract weight statistics for each layer
    for layer, weights in details["weights"].items():
        if isinstance(weights, dict):
            for stat in ["mean", "median", "std", "max", "min", "frobenius_norm", "spectral_norm", "alpha", "alpha_hat"]:#, "var"]:
                record[f"{layer}_{stat}"] = weights.get(stat, None)

    records.append(record)

df = pd.DataFrame(records)

# Remove columns without unique values
# df = df.loc[:, df.nunique() > 1]

# Print metrics
print(df.shape)
# print(df.filter(like="alpha_hat").describe())
# print(df["is_better"].value_counts(normalize=True))

# Print unique values of 'seed'
# print(df["seed"].unique())

# Store the DataFrame in a CSV file
df.to_csv("scripts/experiments/model_stats.csv", index=False)
