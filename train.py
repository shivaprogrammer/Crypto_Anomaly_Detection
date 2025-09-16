import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(model, X_train, X_val,
                epochs=100, batch_size=64, lr=1e-3,
                device="cuda", patience=10, weight_decay=1e-3,
                noise_std=0.05, use_huber=True, save_path=None):
    """
    Train model with:
    - Early stopping
    - Dropout (inside model definition)
    - L2 regularization (weight_decay)
    - Optional Huber loss (robust to outliers)
    - Gradient clipping
    - Input noise injection
    """

    # ✅ Choose loss function
    if use_huber:
        criterion = nn.HuberLoss(delta=1.0)   # smoother than MSE
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        # -------------------
        # Training
        # -------------------
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch = X_train[idx].to(device)

            # 🔹 Add Gaussian noise for regularization
            if noise_std > 0:
                batch = batch + noise_std * torch.randn_like(batch)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()

            # 🔹 Clip gradients to prevent exploding updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(X_train))

        # -------------------
        # Validation
        # -------------------
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, X_val.to(device)).item()
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss={train_losses[-1]:.6f} | "
              f"Val Loss={val_loss:.6f}")

        # -------------------
        # Early Stopping
        # -------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹ Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # -------------------
    # Plot training curve
    # -------------------
    if save_path:
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.savefig(save_path)
        plt.close("all")

    return model, best_val_loss, train_losses, val_losses
