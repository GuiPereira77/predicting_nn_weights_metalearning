import pytorch_lightning as pl
import numpy as np
import weightwatcher as ww
import matplotlib.pyplot as plt
import powerlaw
import uuid
import os

class WeightsPrinterCallback(pl.Callback):
    stats = {}
    ww_metrics = {}

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training to evaluate and analyze weights."""
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

    def _weight_watcher(self, pl_module):
        """Analyzes the model's weight matrices using WeightWatcher."""
        watcher = ww.WeightWatcher()
        describe = watcher.describe(model=pl_module)
        print(describe)
        self.ww_metrics = watcher.get_summary()

    def analyze_weight_matrix(self, W, layer_name):
        """Function to perform SVD and analyze eigenvalues."""
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

        # Plot histogram of eigenvalues
        # self.plot_eigenvalues_histogram(eigenvalues, alpha, fit.xmin, layer_name)

        return {
            "frobenius_norm": float(frobenius_norm),
            "spectral_norm": float(spectral_norm),
            "alpha": float(alpha),
            "alpha_hat": float(alpha_hat)
        }

    def plot_eigenvalues_histogram(self, eigenvalues, alpha, xmin, layer_name):
        """Plots and saves the histogram of eigenvalues with power-law fit."""
        plt.figure(figsize=(6, 4))
        plt.hist(eigenvalues, bins=30, density=True, alpha=0.75, color='b', label='Eigenvalues Histogram')

        if alpha is not None:
            x = np.linspace(min(eigenvalues), max(eigenvalues), 100)
            plt.plot(x, (x / xmin) ** (-alpha), 'r--', label=f'Power Law Fit (alpha={alpha:.2f})')

        plt.xlabel('Eigenvalue')
        plt.ylabel('Density')
        plt.title(f'Histogram of Eigenvalues of W^T W for Layer: {layer_name}')
        plt.legend()

        if not os.path.exists(f'eigen_hist/{layer_name}'):
            os.makedirs(f'eigen_hist/{layer_name}')
        plt.savefig(f'eigen_hist/{layer_name}/{uuid.uuid4()}.png')
        plt.close()

    def print_stats(self):
        """Prints collected weight statistics."""
        for name, stats in self.stats.items():
            print(f"\nLayer: {name}")
            for stat_name, stat_value in stats.items():
                print(f"{stat_name}: {stat_value}")
