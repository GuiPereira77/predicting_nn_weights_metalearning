import os, sys, json
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.callbacks.weights import WeightsPrinterCallback
from src.utils.load_data.config import DATASETS, DATA_GROUPS

# ---- data loading and partitioning
data_name, group = DATA_GROUPS[2]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, test = data_loader.train_test_split(df, horizon=horizon)

# ---- model training
wp_cb = WeightsPrinterCallback()

models = [MLP(input_size=n_lags,
              h=horizon,
              num_layers=3,
              hidden_size=8,
              accelerator='cpu',
              callbacks=[wp_cb],
              max_steps=5)]

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train)
fcst = nf.predict()

cv = fcst.merge(test, on=['unique_id', 'ds'])

# ---- evaluate models
metrics = [smape]
scores = {}

for model in models:
    model_name = model.__class__.__name__
    scores[model_name] = {}
    for metric in metrics:
        metric_name = metric.__name__
        scores_df = evaluate(df=cv, models=[model_name], metrics=[metric])
        scores[model_name][metric_name] = scores_df.mean(numeric_only=True)[model_name]

smape_mean_score = scores['MLP']['smape']

# ---- create model statistics dictionary
results = {}
for model in models:
    key = f"{model}_{data_name}_{group}"
    results[key] = {
        'dataset': {
            'name': data_name,
            'group': group,
        },
        'model': {
            'name': model.__class__.__name__,
            'input_size': model.input_size,
            'horizon': model.h,
            'num_layers': model.num_layers,
            'hidden_size': model.hidden_size,
            'max_steps': model.max_steps,
        },
        'scores': scores[model.__class__.__name__],
        'weights': wp_cb.stats
    }

# ---- save results to a JSON file
output_dir = './scripts/experiments'

# Write the results to a JSON file
output_file = os.path.join(output_dir, 'model_stats.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

