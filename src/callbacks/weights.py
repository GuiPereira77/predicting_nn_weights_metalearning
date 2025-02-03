import pytorch_lightning as pl
import numpy as np

class WeightsPrinterCallback(pl.Callback):
    stats = {}

    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     self._print_weights(pl_module)

    def on_train_end(self, trainer, pl_module):
        # self._print_weights(pl_module)
        self._evaluate_weights(pl_module)

    def _print_weights(self, pl_module):
        state_dict = pl_module.state_dict()
        for name, tensor in state_dict.items():
            if 'weight' in name:
                print(f"\nLayer: {name}")
                print(f"Shape: {tensor.shape}")
                print(f"Values:\n{tensor}")

    def _evaluate_weights(self, pl_module):
        state_dict = pl_module.state_dict()
        for name, tensor in state_dict.items():
            if 'weight' in name:
                tensor_np = tensor.cpu().numpy()
                self.stats[name] = {
                    'shape': tensor_np.shape,
                    'input_size': tensor_np.shape[1],
                    'output_size': tensor_np.shape[0],
                    'mean': np.mean(tensor_np).item(),
                    'median': np.median(tensor_np).item(),
                    'std': np.std(tensor_np).item(),
                    'max': np.max(tensor_np).item(),
                    'min': np.min(tensor_np).item()
                }

    def print_stats(self):
        for name, stats in self.stats.items():
            print(f"\nLayer: {name}")
            for stat_name, stat_value in stats.items():
                print(f"{stat_name}: {stat_value}")
