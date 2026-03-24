# %%
print("Importing PILOT...")
from pilot.pilot import PILOT

print("Import done")
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Callable
from sklearn.neighbors import KernelDensity


# %%
pivot = 4
X_test = np.linspace(-3, 6, 5000).reshape(-1, 1)
def generate_mixture_data(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate X from a mixture of 2 Gaussians: 0.8 N(0,1) + 0.2 N(4,1)
    """
    np.random.seed(seed)

    # Sample from mixture
    mixture_component = np.random.rand(n) < 0.8
    X = np.zeros(n)
    X[mixture_component] = np.random.randn(np.sum(mixture_component))
    X[~mixture_component] = pivot + np.random.randn(np.sum(~mixture_component))

    return X.reshape(-1, 1)

def f_blin(X: np.ndarray) -> np.ndarray:
    """Bilinear (piecewise linear with breakpoint) function"""
    x = X.flatten()
    y = np.zeros_like(x)
    y[x < pivot] = 7 - 1 * x[x < pivot]
    y[x >= pivot] = 1 + 0.5 * x[x >= pivot]
    return y


def f_plin(X: np.ndarray) -> np.ndarray:
    """Piecewise linear function with multiple segments"""
    x = X.flatten()
    y = np.zeros_like(x)
    y[x < pivot] = 7 - 1 * x[x < pivot]
    y[x >= pivot] = 12 - 1.5 * x[x >= pivot]
    return y


# %%
def plot_single_fit(
        true_function: Callable,
        true_name: str,
        n: int = 100,
        noise_level: float = 0.5,
        seed: int = 42,
        output_dir: str = None
):
    """Plot training data, true function, and fitted curve."""
    np.random.seed(seed)

    # Generate training data
    X = generate_mixture_data(n, seed=seed)
    f_true = true_function(X)
    epsilon = np.random.randn(n) * noise_level
    Y = f_true + epsilon

    # Calculate weights
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(X)
    density = np.exp(kde.score_samples(X))
    W = 1.0 / (density + 1e-6)
    W = W / np.mean(W)

    # Fit model
    model_equal = PILOT()
    model_equal.fit(X, Y)

    model_inverse = PILOT()
    model_inverse.fit(X, Y, W)

    # Generate predictions
    y_pred_equal = model_equal.predict(X_test)
    y_pred_inverse = model_inverse.predict(X_test)
    y_true = true_function(X_test)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(X, Y, s=14, color='gray', label='Training data')
    plt.plot(X_test, y_true, 'k-', linewidth=2, label='True function')
    plt.plot(X_test, y_pred_equal, 'r--', linewidth=2, label=f'PILOT (uniform weights)')
    plt.plot(X_test, y_pred_inverse, 'g-.', linewidth=2, label=f'Weighted PILOT\n(inverse density weights)')
    plt.xlabel('Predictor variable')
    plt.ylabel('Response variable')
    plt.title(f'{true_name}')
    plt.legend()
    plt.tight_layout()

    if output_dir:
        file_name = 'synthetic_' + true_name.lower().replace(' ', '_') + '.pdf'
        plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    plt.show()

#%%
output_map = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'output/synthetic_example')
n = 150
noise_level = 2

plot_single_fit(f_plin, 'Piecewise linear true function', n, noise_level, output_dir=output_map)
plot_single_fit(f_blin, 'Broken linear true function', n, noise_level, seed=402, output_dir=output_map)