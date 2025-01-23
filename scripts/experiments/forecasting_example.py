import torch
import pandas as pd
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
              num_layers=3,
              hidden_size=8,
              accelerator='cpu',
              callbacks=[wp_cb],
              max_steps=5)]

nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=train)
fcst = nf.predict()

cv = fcst.merge(test, on=['unique_id', 'ds'])

scores_df = evaluate(df=cv, models=['MLP'], metrics=[smape])
scores_df.mean(numeric_only=True)['MLP']

t = torch.tensor([[0.0816, 0.2906, -0.1490, -0.1516, -0.3093, -0.0171, -0.1702, 0.0117],
                  [-0.0695, 0.1203, 0.3054, 0.1073, -0.1109, -0.1713, -0.3054, 0.3005],
                  [-0.0643, 0.0738, 0.0153, -0.2231, 0.2015, 0.1988, 0.0201, -0.1987],
                  [-0.1502, 0.0021, -0.2020, -0.1973, -0.0541, -0.1018, 0.0841, -0.0956],
                  [-0.0439, -0.1320, -0.3202, 0.3113, 0.0703, 0.3186, -0.0540, -0.0858],
                  [-0.3350, -0.1370, 0.2289, -0.3490, -0.1266, 0.2154, 0.2011, 0.1278],
                  [-0.0673, -0.0244, -0.0650, -0.1275, 0.1501, -0.1204, -0.1958, 0.0829],
                  [-0.2529, -0.0728, 0.1832, 0.1349, 0.3037, -0.2411, 0.0058, -0.1016]])

arr = t.numpy()

summary = {
    'dataset': data_name,
    'mean': arr.mean(),
    'std': arr.std(),
    'mean_0_max': arr.mean(axis=0).max(),
    'acc': scores_df.mean(numeric_only=True)['MLP'],
}

pd.Series(summary)

