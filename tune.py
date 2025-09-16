import optuna
import torch
from train import train_model
from detect import detect_anomalies
from utils import create_sliding_windows, split_train_test
from models.lstm_autoencoder import LSTMAutoencoder
from models.transformer_anomaly import TransformerAnomalyDetector

def objective(trial, features, raw_df, device, model_type="LSTM_AE"):
    window_size = trial.suggest_categorical("window_size", [60, 90, 120])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  

    if model_type == "LSTM_AE":
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)  
        model = LSTMAutoencoder(
            n_features=features.shape[1],
            hidden_size=hidden_size,
            dropout=dropout
        ).to(device)

    else:  
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)  
        model = TransformerAnomalyDetector(
            n_features=features.shape[1],
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

    # Create sliding windows
    X_windows = create_sliding_windows(features, window_size=window_size)
    n_total = features.shape[0]
    n_train = int(0.8 * n_total)
    X_train, X_test, window_end_indices, train_mask = split_train_test(
        X_windows, window_size, n_train
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    model, best_val_loss, _, _ = train_model(
        model, X_train, X_test,
        epochs=50, batch_size=batch_size,
        lr=lr, device=device, patience=5
    )

    # Evaluate anomalies 
    _, _, precision, recall, f1 = detect_anomalies(
        model, X_train, X_test,
        window_end_indices, train_mask,
        raw_df, f"Tuning-{model_type}",
        save_path=None, device=device
    )

    # Store metrics in trial attributes
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("f1", f1)
    return best_val_loss

def run_hyperparameter_search(features, raw_df, device,
                              model_type="LSTM_AE", n_trials=20):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, features, raw_df, device, model_type),
        n_trials=n_trials
    )
    print(f"🔎 Best hyperparameters for {model_type}: {study.best_params}")
    print(f"Best validation loss: {study.best_value}")
    return study
