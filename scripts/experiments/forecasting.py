import os
import sys
import json
import logging
import torch
import copy
import random
from itertools import product
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.callbacks.weights import WeightsPrinterCallback
from src.utils.load_data.config import DATASETS, DATA_GROUPS

# ---- Configure logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# ---- Detect GPU availability ----
device = "gpu" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device.upper()}")
if device == "gpu":
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    torch.set_float32_matmul_precision('high')  # Options: 'high', 'medium', or 'default'

# ---- Variables ----
# ---- Hyperparameter Combinations using itertools.product ----
HIDDEN_SIZE_LIST = [8, 16, 32, 64]
MAX_STEPS_LIST = [500]
NUM_LAYERS_LIST = [3]
LEARNING_RATE_LIST = [1e-3, 5e-4, 1e-4]
BATCH_SIZE_LIST = [16, 32, 64]
SCALER_TYPE_LIST = ['identity', 'standard', 'robust', 'minmax']
SEED_LIST = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

results = {}

# ---- Model Training for Different Data Groups ----
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

    # ---- Model Training for Seasonal Naive ----
    try:
        logger.info("Starting Seasonal Naive model training")
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=freq_int)],
            freq=freq_str
        )

        sf.fit(df=train)
        sfdf = sf.predict(h=horizon)
        sfdf = sfdf.merge(test, on=['unique_id', 'ds'])
        sn_smape_score = float(evaluate(df=sfdf, models=['SeasonalNaive'], metrics=[smape]).mean(numeric_only=True)['SeasonalNaive'])

        logger.info("Seasonal Naive model training completed.")
    except Exception as e:
        logger.error(f"Error during Seasonal Naive model training: {e}")
        sys.exit(1)

    # ---- Generate all possible combinations of the hyperparameters ----
    hyperparameter_combinations = product(
        HIDDEN_SIZE_LIST,MAX_STEPS_LIST,NUM_LAYERS_LIST,LEARNING_RATE_LIST,BATCH_SIZE_LIST,SCALER_TYPE_LIST,SEED_LIST
        )

    # ---- Model Training for Different Hyperparameter Combinations ----
    for hidden_size, max_steps, num_layers, learning_rate, batch_size, scaler_type, seed in hyperparameter_combinations:
        # Set the random seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create a new MLP model with the current hyperparameters
        wp_cb = WeightsPrinterCallback()
        model = MLP(
            input_size=n_lags,
            h=horizon,
            num_layers=num_layers,
            hidden_size=hidden_size,
            accelerator=device,
            callbacks=[wp_cb],
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            scaler_type=scaler_type,
        )

        try:
            logger.info(f"Starting model training with hidden_size={hidden_size},"
                         f"learning_rate={learning_rate}, batch_size={batch_size}, scaler_type={scaler_type}, seed={seed}")
            nf = NeuralForecast(models=[model], freq=freq_str)
            nf.fit(df=train)
            fcst = nf.predict()
            logger.info(f"Model training completed.")
        except Exception as e:
            logger.error(f"Error during training with hidden_size={hidden_size},"
                         f"learning_rate={learning_rate}, batch_size={batch_size}, scaler_type={scaler_type}, seed={seed}: {e}")
            continue

        cv = fcst.merge(sfdf, on=['unique_id', 'ds'])

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
            'r2_score': r2_score_val,
            'sn_smape': sn_smape_score,
            'is_better': smape_score < sn_smape_score
        }

        logger.info(f"Evaluation completed. SMAPE Score: {scores['smape']}")
        logger.info(f"SN SMAPE Score: {scores['sn_smape']}")
        logger.info(f"Is the MLP model better than the Seasonal Naive model? {scores['is_better']}")

        # ---- Create Model Statistics Dictionary ----
        key = f"{data_name}_{group}_hidden_size_{hidden_size}_learning_rate_{learning_rate}_batch_size_{batch_size}_scaler_type_{scaler_type}_seed_{seed}"
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
                "learning_rate": model.learning_rate,
                "batch_size": model.batch_size,
                "scaler_type": scaler_type,
                "total_params": sum(p.numel() for p in model.parameters()),
            },
            "seed": seed,
            "scores": scores,
            "weights": copy.deepcopy(wp_cb.stats),
        }

        logger.info(f"Model statistics dictionary created for {key}")

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
