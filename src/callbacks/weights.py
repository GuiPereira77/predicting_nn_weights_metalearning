import pytorch_lightning as pl
import numpy as np
import weightwatcher as ww
import matplotlib.pyplot as plt
import powerlaw
import uuid
import os
import torch

class WeightsPrinterCallback(pl.Callback):
    stats = {}
    ww_metrics = {}

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training to evaluate and analyze weights."""
        self._get_gradient_norm(pl_module)
        self._get_model_variance(pl_module)
        self._evaluate_weights(pl_module)

    def _evaluate_weights(self, pl_module):
        """Extracts and computes statistical metrics for weight matrices."""
        state_dict = pl_module.state_dict()
        for name, tensor in state_dict.items():
            if 'weight' in name and tensor.ndimension() > 1:
                tensor_np = tensor.detach().cpu().numpy()
                wm_analysis = self.analyze_weight_matrix(tensor_np, name)
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

    def analyze_weight_matrix(self, W, layer_name):
        """
        Analyze the weight matrix of a neural network layer using Singular Value Decomposition (SVD) 
        and power-law fitting.
        Parameters:
        W (numpy.ndarray): The weight matrix to be analyzed.
        layer_name (str): The name of the layer to which the weight matrix belongs.
        Returns:
        dict: A dictionary containing the following metrics:
            - frobenius_norm (float): The Frobenius norm of the weight matrix.
            - spectral_norm (float): The spectral norm (largest singular value) of the weight matrix.
            - alpha (float): The power-law exponent fitted to the eigenvalues of the weight matrix.
            - alpha_hat (float): The adjusted power-law exponent based on the mean and median of the eigenvalues.
        This function performs the following steps:
        1. Computes the Singular Value Decomposition (SVD) of the weight matrix.
        2. Calculates the eigenvalues of the weight matrix.
        3. Computes the Frobenius norm and spectral norm of the weight matrix.
        4. Fits a power-law distribution to the eigenvalues and calculates the power-law exponent.
        5. Returns a dictionary containing the computed metrics.
        Note:
        - If the power-law fitting fails, 'alpha' and 'alpha_hat' will be set to None.
        """
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
    
    def _get_gradient_norm(self, model):
        """
        Compute the norm of the gradients of the model.

        This method calculates the L2 norm of the gradients for all parameters in the model
        and stores the result in the 'gradient_norm' key of the 'stats' dictionary.

        Gradient Norm ≈ 0 → Vanishing gradients
        Gradient Norm is huge (>10 or 100) → Exploding gradients
        Steady gradient norm (~0.01 - 1) → Healthy training

        Args:
            model (torch.nn.Module): The model whose gradients are to be computed.
        """
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        self.stats['gradient_norm'] = total_norm ** 0.5

    def _get_model_variance(self, model):
        """Compute the variance of the model weights."""
        all_weights = torch.cat([param.data.flatten() for param in model.parameters() if len(param.shape) > 1])
        model_variance = torch.var(all_weights)
        self.stats['model_variance'] = model_variance.item()

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
