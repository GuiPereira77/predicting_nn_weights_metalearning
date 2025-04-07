import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
import torch

class WeightsPrinterCallback(pl.Callback):
    stats = {}

    def on_train_start(self, trainer, pl_module):
        """Called at the beginning of training to get initial statistics."""
        self._get_model_variance(pl_module, 'start')

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training to evaluate and analyze weights."""
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
                # print(f"Power Law fit successful: alpha={alpha:.4f}")
            else:
                raise ValueError("No valid eigenvalues for power-law fitting.")
        except Exception as e:
            alpha = None  # Ensure alpha is defined
            alpha_hat = None
            raise RuntimeError(f"Power Law fitting failed: {e}")

        return {
            "frobenius_norm": float(frobenius_norm),
            "spectral_norm": float(spectral_norm),
            "alpha": float(alpha),
            "alpha_hat": float(alpha_hat)
        }

    def _get_model_variance(self, model, stage):
        """Compute the variance of the model weights."""
        all_weights = torch.cat([param.data.flatten() for param in model.parameters() if len(param.shape) > 1])
        model_variance = torch.var(all_weights)
        self.stats['model_variance_'+stage] = model_variance.item()
