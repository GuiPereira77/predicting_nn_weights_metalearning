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
from itertools import product

def configure_logging():
    """ Configure logging settings."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    # logger.setLevel(logging.ERROR)
    return logger

logger = configure_logging()

def detect_device():
    """ Detect the available device (GPU or CPU). """
    device = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device.upper()}")
    if device == "gpu":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        torch.set_float32_matmul_precision('high')  # Options: 'high', 'medium', or 'default'
    return device

device = detect_device()

def load_and_prepare_data(data_name, group):
    """ Load and prepare the dataset. """
    try:
        logger.info(f"Loading dataset: {data_name}, Group: {group}")
        data_loader = DATASETS[data_name]
        df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
        train, test = data_loader.train_test_split(df, horizon=horizon)
        logger.info("Data successfully loaded and split.")
        return train, test, horizon, n_lags, freq_str, freq_int
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def train_seasonal_naive(train, test, freq_str, freq_int, horizon):
    """ Train the Seasonal Naive model and evaluate it. """
    try:
        logger.info("Starting Seasonal Naive model training")
        sf = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq_str)
        sf.fit(df=train)
        sfdf = sf.predict(h=horizon).merge(test, on=['unique_id', 'ds'])
        sn_smape_score = float(evaluate(df=sfdf, models=['SeasonalNaive'], metrics=[smape]).mean(numeric_only=True)['SeasonalNaive'])
        logger.info("Seasonal Naive model training completed.")
        return sfdf, sn_smape_score
    except Exception as e:
        logger.error(f"Error during Seasonal Naive model training: {e}")
        sys.exit(1)

def generate_hyperparameter_combinations():
    """ Generate all combinations of hyperparameters. """
    hyperparameters = {
        "hidden_size": [8, 16, 32, 64],
        "max_steps": [500],
        "num_layers": [3],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [16, 32, 64],
        "scaler_type": ['identity', 'standard', 'robust', 'minmax'],
        "seed": [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021],
    }

    return product(*hyperparameters.values())

def display_model_info(model, seed):
    """ Return the model information as a string. """
    info = [
        # f"Model: {model.__class__.__name__}",
        # f"Input Size: {model.input_size}",
        # f"Horizon: {model.h}",
        # f"Num Layers: {model.num_layers}",
        f"Hidden Size: {model.hidden_size}",
        # f"Max Steps: {model.max_steps}",
        f"Learning Rate: {model.learning_rate}",
        f"Batch Size: {model.batch_size}",
        # f"Scaler Type: {model.scaler_type}",
        # f"Total Params: {sum(p.numel() for p in model.parameters())}",
        f"Current Seed: {seed}"
    ]
    return "\n".join(info)

def train_mlp_models(train, sfdf, hyperparameters, freq_str, n_lags, horizon, sn_smape_score):
    """ Train the MLP model with the given hyperparameters. """
    results = {}
    for hidden_size, max_steps, num_layers, learning_rate, batch_size, scaler_type, seed in hyperparameters:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

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
            logger.info(f"Training MLP: {display_model_info(model, seed)}")
            nf = NeuralForecast(models=[model], freq=freq_str)
            nf.fit(df=train)
            fcst = nf.predict()
            logger.info("Model training completed.")
        except Exception as e:
            logger.error(f"Error during training model {display_model_info(model, seed)}: {e}")
            continue

        cv = fcst.merge(sfdf, on=['unique_id', 'ds'])
        scores = evaluate_model(cv, sn_smape_score)
        key = generate_result_key(data_name, group, hidden_size, learning_rate, batch_size, scaler_type, seed)
        results[key] = create_result_entry(data_name, group, model, scaler_type, seed, scores, wp_cb)
        # print(f"Results for {key}: {results[key]}")
    return results

def evaluate_model(cv, sn_smape_score, model='MLP'):
    """ Evaluate the model using various metrics. """
    mse_score = mean_squared_error(cv['y'], cv[model])
    mae_score = mean_absolute_error(cv['y'], cv[model])
    r2_score_val = r2_score(cv['y'], cv[model])
    smape_score = float(evaluate(df=cv, models=[model], metrics=[smape]).mean(numeric_only=True)[model])
    return {
        'smape': smape_score,
        'mse': mse_score,
        'mae': mae_score,
        'r2_score': r2_score_val,
        'sn_smape': sn_smape_score,
        'is_better': smape_score < sn_smape_score
    }

def generate_result_key(data_name, group, hidden_size, learning_rate, batch_size, scaler_type, seed):
    """ Generate a unique key for the result entry. """
    return f"{data_name}_{group}_hidden_size_{hidden_size}_learning_rate_{learning_rate}_batch_size_{batch_size}_scaler_type_{scaler_type}_seed_{seed}"

def create_result_entry(data_name, group, model, scaler_type, seed, scores, wp_cb):
    """ Create a result entry for the model. """
    return {
        "dataset": {"name": data_name, "group": group},
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
        # "weights": copy.deepcopy(wp_cb.stats),
        "weights": wp_cb.stats.copy(),
    }

def save_results(results, output_file):
    """ Save the results to a JSON file. """
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    results = {}
    for data_name, group in DATA_GROUPS:
        train, test, horizon, n_lags, freq_str, freq_int = load_and_prepare_data(data_name, group)
        sfdf, sn_smape_score = train_seasonal_naive(train, test, freq_str, freq_int, horizon)
        hyperparameter_combinations = generate_hyperparameter_combinations()
        group_results = train_mlp_models(train, sfdf, hyperparameter_combinations, freq_str, n_lags, horizon, sn_smape_score)
        results.update(group_results)

        output_file =  "./scripts/experiments/model_stats.json"
        save_results(results, output_file)