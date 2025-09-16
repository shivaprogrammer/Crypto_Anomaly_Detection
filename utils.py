import numpy as np
import matplotlib.pyplot as plt
import torch

# ================================
# 1. Sliding Window Creation
# ================================
def create_sliding_windows(values, window_size=60, step=1):
    """
    Convert time series into overlapping windows.
    values: np.array of shape (n_samples, n_features)
    returns: np.array of shape (n_windows, window_size, n_features)
    """
    n = values.shape[0]
    windows = []
    for start in range(0, n - window_size + 1, step):
        windows.append(values[start:start + window_size])
    return np.stack(windows)

# ================================
# 2. Train/Test Split
# ================================
def split_train_test(X_windows, window_size, n_train):
    """
    Split windows into train and test sets based on window end index.
    """
    window_end_indices = np.arange(window_size - 1, window_size - 1 + X_windows.shape[0])
    train_mask = window_end_indices < n_train
    X_train = X_windows[train_mask]
    X_test = X_windows[~train_mask]
    return X_train, X_test, window_end_indices, train_mask

# ================================
# 3. Device Selection
# ================================
def get_device():
    """Return cuda if available, else cpu"""
    return "cuda" if torch.cuda.is_available() else "cpu"

# ================================
# 4. Plot Anomalies on Price
# ================================
def plot_anomalies(df, anomaly_indices, coin, save_path=None):
    """
    Plot anomalies on the Close price series.
    df: dataframe with 'Close' column
    anomaly_indices: indices flagged as anomalies
    """
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Close'], label="Close Price", color="blue")
    if len(anomaly_indices) > 0:
        plt.scatter(df.index[anomaly_indices], df['Close'].iloc[anomaly_indices], 
                    color="red", label="Anomaly", s=30)
    plt.title(f"Anomalies in {coin}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ================================
# 5. Plot Reconstruction vs Actual
# ================================
def plot_reconstruction(original_window, reconstructed_window, feature_idx=3, feature_name="Close"):
    """
    Plot original vs reconstructed window for debugging anomalies.
    feature_idx: column index (default=3 -> Close)
    """
    time_axis = np.arange(original_window.shape[0])
    plt.figure(figsize=(12,5))
    plt.plot(time_axis, original_window[:, feature_idx], label=f"Original {feature_name}")
    plt.plot(time_axis, reconstructed_window[:, feature_idx], label=f"Reconstructed {feature_name}", linestyle="--")
    plt.title(f"Reconstruction: {feature_name}")
    plt.legend()
    plt.show()
