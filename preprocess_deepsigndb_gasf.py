"""
DeepSignDB to Asymmetric GASF Preprocessing

Converts signatures from DeepSignDB Development set to asymmetric GASF
representations using speed (magnitude of velocity).

Data formats supported:
- 4 columns: x, y, timestamp, pressure
- 6 columns: x, y, timestamp, azimuth, inclination, pressure
- 7 columns: x, y, timestamp, pen_status, azimuth, inclination, pressure
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional
import warnings

from asymmetric_gasf import compute_asymmetric_gasf, normalize_minmax


def compute_first_order_derivative(series: np.ndarray) -> np.ndarray:
    """
    Compute first-order derivative using central differences.

    For endpoints, uses forward/backward differences.

    Args:
        series: Input time series of shape (M,)

    Returns:
        First-order derivative of shape (M,)
    """
    M = len(series)
    derivative = np.zeros(M)

    # Forward difference for first point
    derivative[0] = series[1] - series[0]

    # Central differences for interior points
    derivative[1:-1] = (series[2:] - series[:-2]) / 2.0

    # Backward difference for last point
    derivative[-1] = series[-1] - series[-2]

    return derivative


def load_signature_stylus(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a stylus signature file from DeepSignDB.

    Handles three formats:
    - 4 columns: x, y, timestamp, pressure
    - 6 columns: x, y, timestamp, azimuth, inclination, pressure
    - 7 columns: x, y, timestamp, pen_status, azimuth, inclination, pressure

    Args:
        file_path: Path to the signature txt file

    Returns:
        Tuple of (data array, num_columns)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # First line is the number of points
    num_points = int(lines[0].strip())

    # Parse data lines
    data = []
    for line in lines[1:num_points + 1]:
        values = line.strip().split()
        data.append([float(v) for v in values])

    data = np.array(data)

    # Verify it's valid data (4, 6, or 7 columns)
    if data.shape[1] not in [4, 6, 7]:
        raise ValueError(f"Expected 4, 6, or 7 columns, got {data.shape[1]}")

    return data, data.shape[1]


def extract_speed_channel(data: np.ndarray) -> np.ndarray:
    """
    Extract speed (magnitude of velocity) from x, y coordinates.

    Speed is computed as sqrt(dx^2 + dy^2) where dx and dy are
    first-order derivatives of x and y.

    Args:
        data: Raw signature data (any format with x, y in columns 0, 1)

    Returns:
        Array of shape (M, 1) with speed values
    """
    x = data[:, 0]
    y = data[:, 1]

    # Compute first-order derivatives
    dx = compute_first_order_derivative(x)
    dy = compute_first_order_derivative(y)

    # Compute speed as magnitude of velocity
    speed = np.sqrt(dx**2 + dy**2)

    return speed.reshape(-1, 1)


def signature_to_asymmetric_gasf(
    data: np.ndarray,
    target_length: Optional[int] = None
) -> np.ndarray:
    """
    Convert signature channels to asymmetric GASF representation.

    Args:
        data: Signature data of shape (M, C) where C is number of channels
        target_length: If provided, resample to this length before GASF conversion

    Returns:
        Asymmetric GASF of shape (C, target_length//2, target_length//2)
    """
    M, C = data.shape

    # Resample to target length if specified
    if target_length is not None and M != target_length:
        resampled = np.zeros((target_length, C))
        for c in range(C):
            resampled[:, c] = np.interp(
                np.linspace(0, M - 1, target_length),
                np.arange(M),
                data[:, c]
            )
        data = resampled

    # Compute asymmetric GASF for each channel
    gasf_channels = []
    for c in range(C):
        gasf = compute_asymmetric_gasf(data[:, c], normalize=True)
        gasf_channels.append(gasf)

    return np.stack(gasf_channels, axis=0)


def find_stylus_signatures(dev_dir: str) -> List[str]:
    """
    Find all stylus signature files in Development set.

    Args:
        dev_dir: Path to Development directory

    Returns:
        List of file paths
    """
    stylus_dir = os.path.join(dev_dir, "stylus")
    signature_files = []

    for file in os.listdir(stylus_dir):
        if file.endswith('.txt'):
            signature_files.append(os.path.join(stylus_dir, file))

    return signature_files


def preprocess_deepsigndb(
    development_dir: str,
    output_path: str,
    target_length: int = 256
):
    """
    Preprocess DeepSignDB Development stylus set to asymmetric GASF representations.
    Uses speed (magnitude of velocity) as the single channel.

    Args:
        development_dir: Path to Development directory
        output_path: Path to save the output .npz file
        target_length: Target sequence length (will be resampled).
                       GASF output will be (target_length//2, target_length//2)
    """
    print("=" * 60)
    print("DeepSignDB to Asymmetric GASF Preprocessing")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Dataset: Development/stylus only")
    print(f"  - Channel: speed (magnitude of velocity)")
    print(f"  - Target length: {target_length}")
    print(f"  - GASF output size: {target_length//2} x {target_length//2}")

    # Find all stylus signature files
    print(f"\nSearching for stylus signatures in: {development_dir}")
    signature_files = find_stylus_signatures(development_dir)
    print(f"Found {len(signature_files)} stylus signature files")

    # Process all signatures
    print(f"\nProcessing signatures...")

    gasf_data = []
    file_names = []
    skipped_signatures = []  # Track skipped signatures with reasons

    for file_path in tqdm(signature_files, desc="Converting to GASF"):
        file_name = os.path.basename(file_path)
        try:
            # Load signature
            data, num_columns = load_signature_stylus(file_path)

            # Skip if too short (less than 10 points)
            if len(data) < 10:
                skipped_signatures.append({
                    'file': file_name,
                    'reason': 'too_short',
                    'details': f'Only {len(data)} points (minimum: 10)'
                })
                continue

            # Extract speed channel
            speed_channel = extract_speed_channel(data)

            # Convert to asymmetric GASF
            gasf = signature_to_asymmetric_gasf(speed_channel, target_length)

            gasf_data.append(gasf)
            file_names.append(file_name)

        except ValueError as e:
            skipped_signatures.append({
                'file': file_name,
                'reason': 'invalid_format',
                'details': str(e)
            })
            continue
        except Exception as e:
            skipped_signatures.append({
                'file': file_name,
                'reason': 'processing_error',
                'details': str(e)
            })
            continue

    # Stack all GASF data
    gasf_data = np.stack(gasf_data, axis=0)

    # Summarize skipped signatures by reason
    skip_reasons = {}
    for item in skipped_signatures:
        reason = item['reason']
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    print(f"\nProcessing complete!")
    print(f"  - Successfully processed: {len(gasf_data)}")
    print(f"  - Skipped: {len(skipped_signatures)}")
    print(f"  - Output shape: {gasf_data.shape}")
    print(f"    (N_samples, N_channels, H, W)")
    print(f"    Channel: [speed]")

    # Print skip reason breakdown
    if skipped_signatures:
        print(f"\n  Skipped signatures breakdown:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"    - {reason}: {count}")

        # Print details of first few skipped signatures for each reason
        print(f"\n  Sample skipped signatures (up to 5 per reason):")
        for reason in sorted(skip_reasons.keys()):
            print(f"\n    [{reason}]:")
            reason_items = [s for s in skipped_signatures if s['reason'] == reason]
            for item in reason_items[:5]:
                print(f"      - {item['file']}: {item['details']}")

    # Save to npz
    print(f"\nSaving to: {output_path}")
    np.savez_compressed(
        output_path,
        gasf_data=gasf_data,
        file_names=np.array(file_names),
        target_length=target_length,
        channels=np.array(['speed'])
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved! File size: {file_size_mb:.2f} MB")

    # Save skipped signatures log
    if skipped_signatures:
        log_path = output_path.replace('.npz', '_skipped.log')
        print(f"\nSaving skipped signatures log to: {log_path}")
        with open(log_path, 'w') as f:
            f.write("Skipped Signatures Log\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total skipped: {len(skipped_signatures)}\n\n")
            f.write("Breakdown by reason:\n")
            for reason, count in sorted(skip_reasons.items()):
                f.write(f"  - {reason}: {count}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("Detailed list:\n\n")
            for item in skipped_signatures:
                f.write(f"File: {item['file']}\n")
                f.write(f"  Reason: {item['reason']}\n")
                f.write(f"  Details: {item['details']}\n\n")
        print(f"Skipped log saved!")

    return gasf_data, skipped_signatures


if __name__ == "__main__":
    # Configuration
    DEVELOPMENT_DIR = r"C:\Users\Himanshu Singhal\Desktop\BTP\DeepSignDB\DeepSignDB\Development"
    OUTPUT_PATH = r"C:\Users\Himanshu Singhal\Desktop\BTP\vit_pretraining\deepsigndb_asymmetric_gasf.npz"
    TARGET_LENGTH = 512  # Sequence length before GASF (output will be 256x256)

    # Run preprocessing
    gasf_data, skipped_signatures = preprocess_deepsigndb(
        development_dir=DEVELOPMENT_DIR,
        output_path=OUTPUT_PATH,
        target_length=TARGET_LENGTH
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"GASF data shape: {gasf_data.shape}")
    print(f"GASF value range: [{gasf_data.min():.4f}, {gasf_data.max():.4f}]")
    print(f"GASF mean: {gasf_data.mean():.4f}")
    print(f"GASF std: {gasf_data.std():.4f}")

    # Per-channel statistics
    print("\nChannel statistics:")
    speed_data = gasf_data[:, 0, :, :]
    print(f"  speed: mean={speed_data.mean():.4f}, std={speed_data.std():.4f}, "
          f"range=[{speed_data.min():.4f}, {speed_data.max():.4f}]")

    # Skipped signatures summary
    if skipped_signatures:
        print("\n" + "=" * 60)
        print("Skipped Signatures Summary")
        print("=" * 60)
        print(f"Total skipped: {len(skipped_signatures)}")
        print(f"Success rate: {len(gasf_data) / (len(gasf_data) + len(skipped_signatures)) * 100:.2f}%")
        print(f"\nFull details saved to: {OUTPUT_PATH.replace('.npz', '_skipped.log')}")
