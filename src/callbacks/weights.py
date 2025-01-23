import copy
import random
from typing import Union, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch


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

#
# class AdvancedWeightsPrinterCallback(pl.Callback):
#     def __init__(self, layers_to_track=None, save_to_file=False, print_freq=1):
#         """
#         Args:
#             layers_to_track: List of layer names to track (None for all layers)
#             save_to_file: Whether to save weights to a file
#             print_freq: Print weights every N epochs
#         """
#         super().__init__()
#         self.layers_to_track = layers_to_track
#         self.save_to_file = save_to_file
#         self.print_freq = print_freq
#
#     def on_epoch_end(self, trainer, pl_module):
#         if (trainer.current_epoch + 1) % self.print_freq == 0:
#             output = [f"\nWeights at epoch {trainer.current_epoch + 1}:"]
#
#             for name, param in pl_module.named_parameters():
#                 if 'weight' in name:
#                     if self.layers_to_track is None or name in self.layers_to_track:
#                         layer_info = [
#                             f"\nLayer: {name}",
#                             f"Shape: {param.shape}",
#                             f"Values:\n{param.data}",
#                             f"Mean: {torch.mean(param.data):.4f}",
#                             f"Std: {torch.std(param.data):.4f}"
#                         ]
#                         output.extend(layer_info)
#
#             # Print to console
#             print('\n'.join(output))
#
#             # Save to file if requested
#             if self.save_to_file:
#                 filename = f"weights_epoch_{trainer.current_epoch + 1}.txt"
#                 with open(filename, 'w') as f:
#                     f.write('\n'.join(output))
#
#
# # Usage example with specific layers:
# callback = AdvancedWeightsPrinterCallback(
#     layers_to_track=['layer1.weight', 'layer2.weight'],
#     save_to_file=True,
#     print_freq=5
# )
#
#
# class WeightsPrinterCallback(pl.Callback):
#     def __init__(self, print_freq=1):
#         """
#         Args:
#             print_freq: Print weights every N epochs
#         """
#         super().__init__()
#         self.print_freq = print_freq
#
#     def on_epoch_end(self, trainer, pl_module):
#         if (trainer.current_epoch + 1) % self.print_freq == 0:
#             print(f"\nWeights at epoch {trainer.current_epoch + 1}:")
#
#             # Iterate through named parameters
#             for name, param in pl_module.named_parameters():
#                 if 'weight' in name:  # Only print weight parameters
#                     print(f"\nLayer: {name}")
#                     print(f"Shape: {param.shape}")
#                     print(f"Values:\n{param.data}")
#                     print(f"Mean: {torch.mean(param.data):.4f}")
#                     print(f"Std: {torch.std(param.data):.4f}")
