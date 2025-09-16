import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import plot_anomalies
from sklearn.metrics import precision_score, recall_score, f1_score

def detect_anomalies(model, X_train, X_test, window_end_indices, train_mask, df, coin, k=95, device="cuda", save_path=None):
    """
    Detect anomalies with percentile threshold and plot reconstruction error distribution.
    """
    model.eval()
    with torch.no_grad():
        recon_train = model(X_train.to(device)).cpu().numpy()
        mse_train = np.mean((X_train.cpu().numpy() - recon_train)**2, axis=(1,2))

        recon_test = model(X_test.to(device)).cpu().numpy()
        mse_test = np.mean((X_test.cpu().numpy() - recon_test)**2, axis=(1,2))

    # Threshold based on training distribution
    threshold = np.percentile(mse_train, k)
    print(f"[{coin}] Threshold (p{k})={threshold:.6f}")

    # Get anomaly indices
    test_window_end_indices = window_end_indices[~train_mask]
    anomaly_mask = mse_test > threshold
    anomaly_indices = test_window_end_indices[anomaly_mask]


    #  Plot reconstruction error distribution
    plt.figure(figsize=(8,5))
    plt.hist(mse_test, bins=50, alpha=0.7, label="Test Errors", color="blue")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({k}%)")
    plt.title(f"Reconstruction Error Distribution - {coin}")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.legend()
    if save_path:
        error_path = save_path.replace(".png", "_errors.png")
        plt.savefig(error_path)
        plt.close("all")
        print(f" Saved error distribution to {error_path}")
    else:
        plt.show()
        plt.close("all")


    # If anomaly labels exist in df
    precision = recall = f1 = None
    if "Anomaly" in df.columns:
        y_true = df["Anomaly"].iloc[test_window_end_indices].values
        y_pred = anomaly_mask.astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"[{coin}] Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    # Plot anomalies on price series
    plot_anomalies(df, anomaly_indices, coin, save_path)

    return anomaly_indices, threshold, precision, recall, f1
