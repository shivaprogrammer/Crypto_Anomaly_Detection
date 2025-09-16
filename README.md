# 🚀 Cryptocurrency Time-Series Anomaly Detection using Deep Learning

This project implements **financial time series anomaly detection** on a multi-coin cryptocurrency dataset using **deep learning**.  
We leverage **LSTM Autoencoders** and **Transformer-based anomaly detectors** to identify unusual market movements in crypto assets.  

The pipeline covers:
- Data preprocessing & feature engineering
- Exploratory data analysis (EDA)
- Sliding-window time series modeling
- Hyperparameter tuning with Optuna
- Training with advanced regularization
- Anomaly detection via reconstruction error
- Result logging and visualization

---

## 📂 Project Structure

```bash
cyptocurrency/
│── data_preprocessing.py   # Preprocess raw crypto CSVs (scaling, returns, train/test split)
│── analysis.py             # Exploratory data analysis (EDA) plots
│── train.py                # Model training with early stopping, Huber loss, dropout, noise
│── detect.py               # Anomaly detection logic (reconstruction error thresholding)
│── tune.py                 # Hyperparameter tuning with Optuna
│── models/
│    ├── lstm_autoencoder.py
│    ├── transformer_anomaly.py
│── utils.py                # Helper functions (sliding windows, plotting, device)
│── main.py                 # Main pipeline (EDA → training → anomaly detection → results)
│── results/                # Saved plots and summary CSV
│── summary.csv             # Final results table


## 📊 Dataset

- Source: Custom multi-coin **cryptocurrency dataset** (23 coins)  
- Format: CSV files with OHLCV data (`Open`, `High`, `Low`, `Close`, `Volume`)  
- Features engineered:
  - Daily returns
  - Log returns
  - Sliding windows of historical prices
  - Scaled features (StandardScaler / MinMaxScaler)

---

## 📈 Exploratory Data Analysis (EDA)

Performed with `analysis.py`:
- 📉 **Volume trends** across coins  
- 💹 **Daily returns vs time**  
- 📈 **Cumulative returns**  
- 📉 **Drawdown curves**  
- 🔥 **Return correlation heatmaps**  
- 🔥 **Volume correlation heatmaps**  
- 📊 **Volatility & Sharpe ratio comparisons**  

---

## 🧠 Models

### 🔹 LSTM Autoencoder
- Encodes time-series windows into a latent representation  
- Reconstructs them → anomalies show **high reconstruction error**  
- Tuned hyperparams: `hidden_size`, `dropout`, `lr`, `batch_size`, `window_size`

### 🔹 Transformer Anomaly Detector
- Attention-based model for capturing long-term dependencies  
- Tuned hyperparams: `d_model`, `n_heads`, `num_layers`, `dropout`

---

## ⚙️ Training Enhancements

- ✅ **Huber Loss** (robust to outliers in financial data)  
- ✅ **Dropout regularization** (0.3–0.5)  
- ✅ **L2 regularization** (weight decay, default `1e-3`)  
- ✅ **Gradient clipping** (avoid exploding gradients)  
- ✅ **Noise injection** (`noise_std=0.05`) → improves generalization  
- ✅ **Early stopping** (patience = 5–10)  

---

## 🔄 Workflow Diagram

```mermaid
flowchart TD
    A[Raw Crypto CSVs] --> B[Data Preprocessing]
    B --> C[Exploratory Data Analysis]
    B --> D[Feature Engineering: Returns, Scaling]
    D --> E[Sliding Window Generation]
    E --> F[LSTM Autoencoder]
    E --> G[Transformer Anomaly Detector]
    F --> H[Training + Early Stopping]
    G --> H
    H --> I[Reconstruction Error Calculation]
    I --> J[Anomaly Detection (Thresholding)]
    J --> K[Evaluation & Summary CSV]

---
## 📊 Results Summary

| Coin           | Model       |   Best_Val_Loss |   Threshold |   Anomalies |
|:---------------|:------------|----------------:|------------:|------------:|
| Cosmos         | LSTM_AE     |        0.489559 |    0.033545 |         152 |
| Cosmos         | Transformer |        0.225756 |    0.000141 |         169 |
| Stellar        | LSTM_AE     |        0.042103 |    0.035534 |         183 |
| Stellar        | Transformer |        0.002979 |    0.000195 |         225 |
| XRP            | LSTM_AE     |        0.027151 |    0.036653 |         224 |
| XRP            | Transformer |        0.001853 |    0.000089 |         235 |
| CryptocomCoin  | LSTM_AE     |        0.051069 |    0.022215 |         154 |
| CryptocomCoin  | Transformer |        0.015070 |    0.000216 |         180 |
| WrappedBitcoin | LSTM_AE     |        0.157626 |    0.004204 |         178 |
| WrappedBitcoin | Transformer |        0.042120 |    0.000493 |         178 |
| Ethereum       | LSTM_AE     |        0.113523 |    0.031734 |         180 |
| Ethereum       | Transformer |        0.027051 |    0.000343 |         185 |
| Monero         | LSTM_AE     |        0.125606 |    0.023639 |         300 |
| Monero         | Transformer |        0.067336 |    0.000409 |         300 |
| USDCoin        | LSTM_AE     |        0.015268 |    0.021331 |         135 |
| USDCoin        | Transformer |        0.000142 |    0.000121 |          82 |
| BinanceCoin    | LSTM_AE     |        1.741040 |    0.035047 |         156 |
| BinanceCoin    | Transformer |        1.550140 |    0.000279 |         186 |
| NEM            | LSTM_AE     |        0.034726 |    0.040839 |         251 |
| NEM            | Transformer |        0.010210 |    0.000137 |         251 |
| Aave           | LSTM_AE     |        0.035769 |    0.086993 |          27 |
| Aave           | Transformer |        0.000330 |    0.000290 |          55 |
| Uniswap        | LSTM_AE     |        0.020750 |    0.054162 |          16 |
| Uniswap        | Transformer |        0.000223 |    0.000175 |          44 |
| Polkadot       | LSTM_AE     |        0.023642 |    0.063270 |          21 |
| Polkadot       | Transformer |        0.000237 |    0.000201 |          46 |
| Solana         | LSTM_AE     |        0.028632 |    0.051832 |          28 |
| Solana         | Transformer |        0.000250 |    0.000188 |          53 |
| ChainLink      | LSTM_AE     |        0.026843 |    0.057936 |          22 |
| ChainLink      | Transformer |        0.000236 |    0.000209 |          48 |
| Cardano        | LSTM_AE     |        0.030851 |    0.049871 |          25 |
| Cardano        | Transformer |        0.000242 |    0.000195 |          50 |
| Tron           | LSTM_AE     |        0.033589 |    0.047321 |          26 |
| Tron           | Transformer |        0.000251 |    0.000201 |          52 |
| NEM            | LSTM_AE     |        0.034726 |    0.040839 |         251 |
| NEM            | Transformer |        0.010210 |    0.000137 |         251 |
| Bitcoin        | LSTM_AE     |        0.121972 |    0.004936 |         185 |
| Bitcoin        | Transformer |        0.027334 |    0.000295 |         196 |
| Litecoin       | LSTM_AE     |        0.068229 |    0.016231 |         192 |
| Litecoin       | Transformer |        0.012781 |    0.000241 |         207 |
| Iota           | LSTM_AE     |        0.045872 |    0.029332 |         201 |
| Iota           | Transformer |        0.009726 |    0.000223 |         216 |
| EOS            | LSTM_AE     |        0.050231 |    0.025678 |         195 |
| EOS            | Transformer |        0.011032 |    0.000227 |         209 |
| Tether         | LSTM_AE     |        0.020562 |    0.018765 |          50 |
| Tether         | Transformer |        0.005431 |    0.000132 |          73 |

