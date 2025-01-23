import pytorch_lightning as pl


class WeightsPrinterCallback(pl.Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._print_weights(pl_module)

    def _print_weights(self, pl_module):
        state_dict = pl_module.state_dict()
        for name, tensor in state_dict.items():
            if 'weight' in name:
                print(f"\nLayer: {name}")
                print(f"Shape: {tensor.shape}")
                print(f"Values:\n{tensor}")

        print('\n---------\n')
