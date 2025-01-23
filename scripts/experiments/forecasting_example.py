from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape

from src.callbacks.weights import WeightsPrinterCallback

from src.utils.load_data.config import DATASETS, DATA_GROUPS

# ---- data loading and partitioning
data_name, group = DATA_GROUPS[2]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

train, test = data_loader.train_test_split(df, horizon=horizon)

wp_cb = WeightsPrinterCallback()

# models = [MLP(input_size=horizon, h=horizon, callbacks=[augmentation_cb])]
models = [MLP(input_size=n_lags,
              h=horizon,
              num_layers=1,
              hidden_size=16,
              accelerator='cpu',
              callbacks=[wp_cb],
              max_steps=5)]

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train)
fcst = nf.predict()

cv = fcst.merge(test, on=['unique_id', 'ds'])

scores_df = evaluate(df=cv, models=['MLP'], metrics=[smape])
scores_df.mean(numeric_only=True)
