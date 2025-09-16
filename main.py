import os
import torch
import pandas as pd
from data_preprocessing import prepare_features, load_all_coins
from analysis import (
    plot_volume, plot_returns, plot_cumulative_returns,
    plot_drawdown, plot_return_correlation, plot_volume_correlation,
    compute_risk_metrics, plot_volatility_and_sharpe
)
from utils import create_sliding_windows, split_train_test, get_device, plot_reconstruction
from train import train_model
from detect import detect_anomalies
from models.lstm_autoencoder import LSTMAutoencoder
from models.transformer_anomaly import TransformerAnomalyDetector
from tune import run_hyperparameter_search


# ========================
# CONFIG
# ========================
DATA_PATH = "/data/datasets/data/crypto/all_data"
RESULTS_PATH = "/data/m24csa029/MTP/cyptocurrency/results"
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
COMPARE_MODE = True

os.makedirs(RESULTS_PATH, exist_ok=True)
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ========================
# LOAD DATA
# ========================
print("\n📂 Loading all coins...")
dataframes = load_all_coins(DATA_PATH)
print(f"✅ Loaded {len(dataframes)} coins")

# ========================
# RUN ANALYSIS
# ========================
plot_volume(dataframes)
plot_returns(dataframes)
plot_cumulative_returns(dataframes)
plot_return_correlation(dataframes)
plot_volume_correlation(dataframes)
metrics = compute_risk_metrics(dataframes)
print(metrics)
plot_volatility_and_sharpe(metrics)

if "coin_Bitcoin" in dataframes:
    plot_drawdown(dataframes["coin_Bitcoin"], "Bitcoin")

# ========================
# TRAIN + ANOMALY DETECTION
# ========================
summary = []

for coin, raw_df in dataframes.items():
    print(f"\n🚀 Processing {coin}...")

    # Preprocess
    df_scaled, scaler = prepare_features(raw_df, test_size=0.2, scaler_type="minmax")
    features = df_scaled.values
    n_total = features.shape[0]
    n_train = int((1 - TEST_SIZE) * n_total)

    if COMPARE_MODE:
        # 🔎 Hyperparameter search
        print(f"\n🔎 Running hyperparameter search for LSTM_AE on {coin}...")
        lstm_study = run_hyperparameter_search(features, raw_df, device,
                                               model_type="LSTM_AE", n_trials=10)

        print(f"\n🔎 Running hyperparameter search for Transformer on {coin}...")
        trans_study = run_hyperparameter_search(features, raw_df, device,
                                                model_type="Transformer", n_trials=10)

        best_lstm_params = lstm_study.best_params
        best_trans_params = trans_study.best_params

        print(f"✅ Best LSTM params: {best_lstm_params}")
        print(f"✅ Best Transformer params: {best_trans_params}")

        models = {}

        # ------------------------
        # 🔹 Build LSTM AE with its best window size
        # ------------------------
        lstm_window = best_lstm_params["window_size"]
        X_windows = create_sliding_windows(features, window_size=lstm_window)
        X_train, X_test, window_end_indices, train_mask = split_train_test(X_windows, lstm_window, n_train)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

        models["LSTM_AE"] = LSTMAutoencoder(
            n_features=features.shape[1],
            hidden_size=best_lstm_params["hidden_size"],
            dropout=best_lstm_params["dropout"]
        ).to(device)

        # ------------------------
        # 🔹 Build Transformer with its best window size
        # ------------------------
        trans_window = best_trans_params["window_size"]
        X_windows_trans = create_sliding_windows(features, window_size=trans_window)
        X_train_trans, X_test_trans, window_end_indices_trans, train_mask_trans = split_train_test(
            X_windows_trans, trans_window, n_train
        )
        X_train_trans = torch.tensor(X_train_trans, dtype=torch.float32).to(device)
        X_test_trans = torch.tensor(X_test_trans, dtype=torch.float32).to(device)

        models["Transformer"] = TransformerAnomalyDetector(
            n_features=features.shape[1],
            d_model=best_trans_params["d_model"],
            n_heads=best_trans_params["n_heads"],
            num_layers=best_trans_params["num_layers"],
            dropout=best_trans_params["dropout"]
        ).to(device)

        # Store separate datasets for each model
        model_datasets = {
            "LSTM_AE": (X_train, X_test, window_end_indices, train_mask),
            "Transformer": (X_train_trans, X_test_trans, window_end_indices_trans, train_mask_trans),
        }

    else:
        models = {"LSTM_AE": LSTMAutoencoder(n_features=features.shape[1]).to(device)}
        X_windows = create_sliding_windows(features, window_size=90)
        X_train, X_test, window_end_indices, train_mask = split_train_test(X_windows, 90, n_train)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        model_datasets = {"LSTM_AE": (X_train, X_test, window_end_indices, train_mask)}

    # ========================
    # Train & Detect
    # ========================
    for model_name, model in models.items():
        print(f"\n>>> Training {model_name} for {coin}...")

        # dataset for this model
        X_train, X_test, window_end_indices, train_mask = model_datasets[model_name]

        # Save curve path
        save_curve = os.path.join(RESULTS_PATH, f"{coin}_{model_name}_training_curve.png")

        # Train
        model, best_val_loss, train_losses, val_losses = train_model(
            model, X_train, X_test,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=device,
            patience=7, weight_decay=1e-4, save_path=save_curve, noise_std=0.01
        )

        # Detect anomalies
        save_plot = os.path.join(RESULTS_PATH, f"{coin}_{model_name}_anomalies.png")
        anomaly_indices, threshold, precision, recall, f1 = detect_anomalies(
            model, X_train, X_test, window_end_indices, train_mask,
            raw_df, f"{coin} ({model_name})", save_path=save_plot, device=device
        )

        print(f"{coin} [{model_name}]: {len(anomaly_indices)} anomalies detected")

        summary.append({
            "Coin": coin,
            "Model": model_name,
            "Best_Val_Loss": best_val_loss,
            "Threshold": threshold,
            "Anomalies": len(anomaly_indices),
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

# ========================
# SAVE SUMMARY CSV
# ========================
summary_df = pd.DataFrame(summary)
summary_csv = os.path.join(RESULTS_PATH, "summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\n✅ Summary saved to {summary_csv}")
