"""
Asymmetric GASF (Gramian Angular Summation Field) Implementation

This module converts time series signature data to asymmetric GASF representations.
The approach:
1. Divide input sequence of length M into two parts (each M/2)
2. Compute symmetric GASF matrices for each part (size M/2 × M/2)
3. Combine the upper triangular half from the first part's GASF with
   the lower triangular half from the second part's GASF
4. Result: Asymmetric GASF of size M/2 × M/2
"""

import numpy as np
from typing import Tuple, Optional


def normalize_minmax(series: np.ndarray, feature_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
    """
    Normalize time series to a given range using min-max scaling.

    Args:
        series: Input time series of shape (M,)
        feature_range: Target range for normalization (default: (-1, 1) for GASF)

    Returns:
        Normalized time series
    """
    min_val, max_val = feature_range
    series_min = np.min(series)
    series_max = np.max(series)

    # Handle constant series
    if series_max - series_min == 0:
        return np.full_like(series, (min_val + max_val) / 2, dtype=np.float64)

    # Scale to [0, 1] then to [min_val, max_val]
    scaled = (series - series_min) / (series_max - series_min)
    return scaled * (max_val - min_val) + min_val


def compute_gasf(series: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric Gramian Angular Summation Field (GASF) matrix.

    GASF is computed as:
        G[i,j] = cos(phi_i + phi_j)
    where phi_i = arccos(x_i) for normalized x_i in [-1, 1]

    Using the identity: cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
    This becomes: G[i,j] = x_i * x_j - sqrt(1-x_i^2) * sqrt(1-x_j^2)

    Args:
        series: Normalized time series in range [-1, 1] of shape (N,)

    Returns:
        GASF matrix of shape (N, N)
    """
    # Clip values to [-1, 1] to avoid numerical issues with arccos
    series = np.clip(series, -1, 1)

    # Compute angular representation
    phi = np.arccos(series)

    # Compute GASF using outer sum of angles
    # G[i,j] = cos(phi_i + phi_j)
    gasf = np.cos(np.add.outer(phi, phi))

    return gasf


def compute_asymmetric_gasf(series: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute the asymmetric GASF representation.

    Process:
    1. Divide input series of length M into two halves (each M/2)
    2. Compute symmetric GASF for each half (size M/2 × M/2)
    3. Take upper triangular part from first half's GASF
    4. Take lower triangular part from second half's GASF
    5. Combine into asymmetric matrix of size M/2 × M/2

    Args:
        series: Input time series of shape (M,). M should be even.
        normalize: Whether to normalize the series to [-1, 1] (default: True)

    Returns:
        Asymmetric GASF matrix of shape (M/2, M/2)
    """
    M = len(series)

    # Ensure even length
    if M % 2 != 0:
        # Pad with last value if odd length
        series = np.append(series, series[-1])
        M = len(series)

    half_length = M // 2

    # Split into two halves
    first_half = series[:half_length]
    second_half = series[half_length:]

    # Normalize each half independently if requested
    if normalize:
        first_half = normalize_minmax(first_half, feature_range=(-1, 1))
        second_half = normalize_minmax(second_half, feature_range=(-1, 1))

    # Compute GASF for each half
    gasf_first = compute_gasf(first_half)   # Shape: (M/2, M/2)
    gasf_second = compute_gasf(second_half)  # Shape: (M/2, M/2)

    # Create asymmetric GASF by combining triangular halves
    # Upper triangular (including diagonal) from first half
    # Lower triangular (excluding diagonal) from second half
    asymmetric_gasf = np.triu(gasf_first) + np.tril(gasf_second, k=-1)

    return asymmetric_gasf


def compute_asymmetric_gasf_multichannel(
    data: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute asymmetric GASF for multi-channel time series data.

    Args:
        data: Input array of shape (M, C) where M is sequence length and C is number of channels
        normalize: Whether to normalize each channel to [-1, 1]

    Returns:
        Stacked asymmetric GASF matrices of shape (C, M/2, M/2)
    """
    if data.ndim == 1:
        # Single channel
        return compute_asymmetric_gasf(data, normalize)[np.newaxis, ...]

    M, C = data.shape
    half_length = (M + 1) // 2  # Account for potential padding

    gasf_images = []
    for channel_idx in range(C):
        channel_data = data[:, channel_idx]
        gasf = compute_asymmetric_gasf(channel_data, normalize)
        gasf_images.append(gasf)

    return np.stack(gasf_images, axis=0)


def batch_compute_asymmetric_gasf(
    batch: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute asymmetric GASF for a batch of time series.

    Args:
        batch: Input array of shape (B, M) for single channel or (B, M, C) for multi-channel
               where B is batch size, M is sequence length, C is number of channels
        normalize: Whether to normalize each series

    Returns:
        Batch of asymmetric GASF matrices
        - Shape (B, 1, M/2, M/2) for single channel input
        - Shape (B, C, M/2, M/2) for multi-channel input
    """
    results = []

    for i in range(len(batch)):
        sample = batch[i]
        if sample.ndim == 1:
            gasf = compute_asymmetric_gasf(sample, normalize)[np.newaxis, ...]
        else:
            gasf = compute_asymmetric_gasf_multichannel(sample, normalize)
        results.append(gasf)

    return np.stack(results, axis=0)


# ============================================================================
# Visualization utilities
# ============================================================================

def visualize_asymmetric_gasf(
    series: np.ndarray,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
):
    """
    Visualize the asymmetric GASF transformation process.

    Shows:
    1. Original time series split into two halves
    2. Symmetric GASF of first half
    3. Symmetric GASF of second half
    4. Final asymmetric GASF

    Args:
        series: Input time series
        figsize: Figure size
        cmap: Colormap for GASF plots
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt

    M = len(series)
    if M % 2 != 0:
        series = np.append(series, series[-1])
        M = len(series)

    half_length = M // 2

    # Normalize and split
    first_half = normalize_minmax(series[:half_length])
    second_half = normalize_minmax(series[half_length:])

    # Compute GASF matrices
    gasf_first = compute_gasf(first_half)
    gasf_second = compute_gasf(second_half)
    asymmetric_gasf = compute_asymmetric_gasf(series, normalize=True)

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Plot 1: Original time series with split
    axes[0].plot(range(half_length), series[:half_length], 'b-', label='First half', linewidth=2)
    axes[0].plot(range(half_length, M), series[half_length:], 'r-', label='Second half', linewidth=2)
    axes[0].axvline(x=half_length, color='k', linestyle='--', alpha=0.5)
    axes[0].set_title('Original Time Series (Split)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: GASF of first half
    im1 = axes[1].imshow(gasf_first, cmap=cmap, aspect='equal', vmin=-1, vmax=1)
    axes[1].set_title('GASF (First Half)\nUpper triangle used')
    axes[1].set_xlabel('Time index')
    axes[1].set_ylabel('Time index')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 3: GASF of second half
    im2 = axes[2].imshow(gasf_second, cmap=cmap, aspect='equal', vmin=-1, vmax=1)
    axes[2].set_title('GASF (Second Half)\nLower triangle used')
    axes[2].set_xlabel('Time index')
    axes[2].set_ylabel('Time index')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Plot 4: Asymmetric GASF
    im3 = axes[3].imshow(asymmetric_gasf, cmap=cmap, aspect='equal', vmin=-1, vmax=1)
    axes[3].set_title('Asymmetric GASF\n(Combined)')
    axes[3].set_xlabel('Time index')
    axes[3].set_ylabel('Time index')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return fig


# ============================================================================
# Example usage and testing
# ============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    print("=" * 60)
    print("Asymmetric GASF - Example Usage")
    print("=" * 60)

    # Create a sample time series (simulating signature data)
    np.random.seed(42)
    M = 100  # Sequence length
    t = np.linspace(0, 4 * np.pi, M)
    sample_series = np.sin(t) + 0.5 * np.sin(2 * t) + 0.1 * np.random.randn(M)

    print(f"\nInput series shape: {sample_series.shape}")
    print(f"Input series length (M): {M}")

    # Compute asymmetric GASF
    asymmetric_gasf = compute_asymmetric_gasf(sample_series)

    print(f"\nAsymmetric GASF shape: {asymmetric_gasf.shape}")
    print(f"Expected shape: ({M//2}, {M//2})")
    print(f"GASF value range: [{asymmetric_gasf.min():.4f}, {asymmetric_gasf.max():.4f}]")

    # Verify asymmetry
    is_symmetric = np.allclose(asymmetric_gasf, asymmetric_gasf.T)
    print(f"\nIs the result symmetric? {is_symmetric}")
    print("(Should be False for asymmetric GASF)")

    # Multi-channel example
    print("\n" + "=" * 60)
    print("Multi-channel Example")
    print("=" * 60)

    # Simulate multi-channel signature data (e.g., x, y, pressure)
    C = 3  # Number of channels
    multichannel_data = np.random.randn(M, C)

    print(f"\nInput multi-channel shape: {multichannel_data.shape}")

    multichannel_gasf = compute_asymmetric_gasf_multichannel(multichannel_data)

    print(f"Multi-channel asymmetric GASF shape: {multichannel_gasf.shape}")
    print(f"Expected shape: ({C}, {M//2}, {M//2})")

    # Batch example
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    B = 5  # Batch size
    batch_data = np.random.randn(B, M, C)

    print(f"\nInput batch shape: {batch_data.shape}")

    batch_gasf = batch_compute_asymmetric_gasf(batch_data)

    print(f"Batch asymmetric GASF shape: {batch_gasf.shape}")
    print(f"Expected shape: ({B}, {C}, {M//2}, {M//2})")

    # Visualization (uncomment to run)
    # print("\n" + "=" * 60)
    # print("Generating Visualization...")
    # print("=" * 60)
    # visualize_asymmetric_gasf(sample_series, save_path="asymmetric_gasf_example.png")
