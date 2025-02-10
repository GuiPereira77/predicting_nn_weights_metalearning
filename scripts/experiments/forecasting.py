import os
import sys
import json
import logging
import torch
import copy
from itertools import product
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.callbacks.weights import WeightsPrinterCallback
from src.utils.load_data.config import DATASETS, DATA_GROUPS

# ---- Configure logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Detect GPU availability ----
device = "gpu" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device.upper()}")

results = {}

for data_name, group in DATA_GROUPS:
    # ---- Data Loading and Preparation ----
    try:
        logger.info(f"Loading dataset: {data_name}, Group: {group}")
        data_loader = DATASETS[data_name]

        df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
        train, test = data_loader.train_test_split(df, horizon=horizon)
        logger.info("Data successfully loaded and split.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # ---- Hyperparameter Combinations using itertools.product ----
    # num_layers_list = [1, 2, 3, 5, 8, 12]   # Example choices for num_layers
    hidden_size_list = [4, 8, 12, 16, 20, 24]  # Example choices for hidden_size
    max_steps_list = [100, 200, 300, 500]  # Example choices for max_steps

    # Generate all possible combinations of the hyperparameters
    hyperparameter_combinations = product(hidden_size_list, max_steps_list)

    # ---- Model Training for Different Hyperparameter Combinations ----
    for hidden_size, max_steps in hyperparameter_combinations:
        # Create a new MLP model with the current hyperparameters
        wp_cb = WeightsPrinterCallback()
        model = MLP(
            input_size=n_lags,
            h=horizon,
            num_layers=3,
            hidden_size=hidden_size,
            accelerator=device,
            callbacks=[wp_cb],
            max_steps=max_steps,
        )

        try:
            logger.info(f"Starting model training with hidden_size={hidden_size}, max_steps={max_steps}...")
            nf = NeuralForecast(models=[model], freq=freq_str)
            nf.fit(df=train)
            fcst = nf.predict()
            logger.info(f"Model training completed for hidden_size={hidden_size}, max_steps={max_steps}.")
        except Exception as e:
            logger.error(f"Error during training with hidden_size={hidden_size}, max_steps={max_steps}: {e}")
            continue

        # TODO: Add training loss calculation

        cv = fcst.merge(test, on=['unique_id', 'ds'])

        # ---- Model Evaluation ----
        # Metrics calculation
        mse_score = mean_squared_error(cv['y'], cv['MLP'])
        mae_score = mean_absolute_error(cv['y'], cv['MLP'])
        r2_score_val = r2_score(cv['y'], cv['MLP'])
        smape_score = float(evaluate(df=cv, models=['MLP'], metrics=[smape]).mean(numeric_only=True)['MLP'])

        # Scores dictionary
        scores = {
            'smape': smape_score,
            'mse': mse_score,
            'mae': mae_score,
            'r2_score': r2_score_val
        }

        logger.info(f"Evaluation completed. SMAPE Score: {scores['smape']}")

        # ---- Create Model Statistics Dictionary ----
        key = f"{data_name}_{group}_hidden_size_{hidden_size}_max_steps_{max_steps}"
        results[key] = {
            "dataset": {
                "name": data_name,
                "group": group,
            },
            "model": {
                "name": model.__class__.__name__,
                "input_size": model.input_size,
                "horizon": model.h,
                "num_layers": model.num_layers,
                "hidden_size": model.hidden_size,
                "max_steps": model.max_steps,
                "total_params": sum(p.numel() for p in model.parameters()),
            },
            "scores": scores,
            # "training_loss": training_loss,
            "weights": copy.deepcopy(wp_cb.stats),
        }

    # ---- Save Results to JSON File ----
    output_dir = "./scripts/experiments"
    output_file = os.path.join(output_dir, "model_stats.json")

    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        sys.exit(1)
