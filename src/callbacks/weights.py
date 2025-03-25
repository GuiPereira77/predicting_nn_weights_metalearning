import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import uuid
import os
import torch

class WeightsPrinterCallback(pl.Callback):
    stats = {}

    def on_train_start(self, trainer, pl_module):
        """Called at the beginning of training to get initial statistics."""
        self._get_model_variance(pl_module, 'start')

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training to evaluate and analyze weights."""
        # self._analyze_model_weights(pl_module)
        # self._get_gradient_norm(pl_module)
        self._get_model_variance(pl_module, 'end')
        self._evaluate_weights(pl_module)

    def _evaluate_weights(self, pl_module):
        """Extracts and computes statistical metrics for weight matrices."""
        state_dict = pl_module.state_dict()
        for name, tensor in state_dict.items():
            if 'weight' in name and tensor.ndimension() > 1:
                tensor_np = tensor.detach().cpu().numpy()
                wm_analysis = self.analyze_weight_matrix(tensor_np)
                self.stats[name] = {
                    'shape': tensor_np.shape,
                    'input_size': tensor_np.shape[1],
                    'output_size': tensor_np.shape[0],
                    'mean': np.mean(tensor_np).item(),
                    'median': np.median(tensor_np).item(),
                    'std': np.std(tensor_np).item(),
                    'max': np.max(tensor_np).item(),
                    'min': np.min(tensor_np).item(),
                    'var': np.var(tensor_np).item(),
                    **wm_analysis
                }

    def analyze_weight_matrix(self, W):
        """ Analyze the weight matrix of the model. """
        # Perform Singular Value Decomposition (SVD)
        U, S, VT = np.linalg.svd(W, full_matrices=False)

        # Eigenvalues of W^T W are the squares of singular values
        eigenvalues = S**2
        # eigenvalues = np.linalg.eigvalsh(W.T @ W)

        # Norm-based metrics
        frobenius_norm = np.linalg.norm(W, 'fro')  # Frobenius norm
        spectral_norm = np.max(S)  # Spectral norm (largest singular value)

        # Power Law fitting with error handling
        try:
            if len(eigenvalues) > 0:
                fit = powerlaw.Fit(eigenvalues, xmin=np.min(eigenvalues))  # Automatic estimation
                alpha = fit.alpha
                alpha_hat = alpha * (np.mean(eigenvalues) / np.median(eigenvalues))
                print(f"Power Law fit successful: alpha={alpha:.4f}")
            else:
                raise ValueError("No valid eigenvalues for power-law fitting.")
        except Exception as e:
            alpha = None  # Ensure alpha is defined
            alpha_hat = None
            print(f"Power Law fit failed: {e}")

        return {
            "frobenius_norm": float(frobenius_norm),
            "spectral_norm": float(spectral_norm),
            "alpha": float(alpha),
            "alpha_hat": float(alpha_hat)
        }

    def _analyze_model_weights(self, model):
        """
        Analyze the combined weight matrix of an entire neural network model.

        Parameters:
        - model (torch.nn.Module): The PyTorch model to analyze.

        Returns:
        - None: The analysis results are stored in the 'stats' dictionary.
        """
        # TODO: Finish this method.
        # Collect all weight matrices
        weight_matrices = []

        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:  # Ensure it's a weight matrix
                W = param.detach().cpu().numpy()
                weight_matrices.append(W)  # Store raw weight matrices

        if len(weight_matrices) == 0:
            raise ValueError("No weight matrices found in the model.")

        # Concatenate all weights into one large matrix
        W_combined = np.vstack([W.reshape(W.shape[0], -1) for W in weight_matrices])  # Reshape each to (rows, flattened cols)
        dictionary = self.analyze_weight_matrix(W_combined)

        # Store the results in the 'stats' dictionary
        self.stats['froebeinus_norm'] = dictionary['frobenius_norm']
        self.stats['spectral_norm'] = dictionary['spectral_norm']
        self.stats['alpha'] = dictionary['alpha']
        self.stats['alpha_hat'] = dictionary['alpha_hat']

    def _get_gradient_norm(self, model):
        """
        Compute the norm of the gradients of the model.

        This method calculates the L2 norm of the gradients for all parameters in the model
        and stores the result in the 'gradient_norm' key of the 'stats' dictionary.

        Gradient Norm ≈ 0 → Vanishing gradients
        Gradient Norm is huge (>10 or 100) → Exploding gradients
        Steady gradient norm (~0.01 - 1) → Healthy training

        Args:
        - model (torch.nn.Module): The model whose gradients are to be computed.

        Returns:
        - None: The gradient norm is stored in the 'stats' dictionary.
        """
        total_norm = 0
        for p in model.parameters():
            if p.grad is None: continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        self.stats['gradient_norm'] = total_norm ** 0.5

    def _get_model_variance(self, model, stage):
        """Compute the variance of the model weights."""
        all_weights = torch.cat([param.data.flatten() for param in model.parameters() if len(param.shape) > 1])
        model_variance = torch.var(all_weights)
        self.stats['model_variance_'+stage] = model_variance.item()

    # def plot_eigenvalues_histogram(self, eigenvalues, alpha, xmin, layer_name):
    #     """Plots and saves the histogram of eigenvalues with power-law fit."""
    #     plt.figure(figsize=(6, 4))
    #     plt.hist(eigenvalues, bins=30, density=True, alpha=0.75, color='b', label='Eigenvalues Histogram')

    #     if alpha is not None:
    #         x = np.linspace(min(eigenvalues), max(eigenvalues), 100)
    #         plt.plot(x, (x / xmin) ** (-alpha), 'r--', label=f'Power Law Fit (alpha={alpha:.2f})')

    #     plt.xlabel('Eigenvalue')
    #     plt.ylabel('Density')
    #     plt.title(f'Histogram of Eigenvalues of W^T W for Layer: {layer_name}')
    #     plt.legend()

    #     if not os.path.exists(f'eigen_hist/{layer_name}'):
    #         os.makedirs(f'eigen_hist/{layer_name}')
    #     plt.savefig(f'eigen_hist/{layer_name}/{uuid.uuid4()}.png')
    #     plt.close()

    # def print_stats(self):
    #     """Prints collected weight statistics."""
    #     for name, stats in self.stats.items():
    #         print(f"\nLayer: {name}")
    #         for stat_name, stat_value in stats.items():
    #             print(f"{stat_name}: {stat_value}")
