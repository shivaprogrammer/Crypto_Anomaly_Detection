import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# =======================
# Helper: ensure 'return' exists
# =======================
def ensure_return(df):
    """Compute daily returns if not already present."""
    if "return" not in df.columns:
        df["return"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    return df

# =======================
# 1. Volume over time
# =======================
def plot_volume(dataframes):
    plt.figure(figsize=(12, 6))
    for name, df in dataframes.items():
        if "Volume" not in df.columns:
            continue
        plt.plot(df.index, df["Volume"], label=name)
    plt.title("Volume across time for all coins")
    plt.xlabel("Date")
    plt.ylabel("Volume (log-scaled)")
    plt.legend()
    plt.show()

# =======================
# 2. Returns over time
# =======================
def plot_returns(dataframes):
    plt.figure(figsize=(12, 6))
    for name, df in dataframes.items():
        df = ensure_return(df)
        plt.plot(df.index, df["return"], label=name)
    plt.title("Daily Returns across coins")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.show()

# =======================
# 3. Cumulative Returns
# =======================
def plot_cumulative_returns(dataframes):
    plt.figure(figsize=(12, 6))
    for name, df in dataframes.items():
        df = ensure_return(df)
        cum_return = (1 + df["return"]).cumprod() - 1
        plt.plot(df.index, cum_return, label=name)
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()

# =======================
# 4. Drawdown for one coin
# =======================
def compute_drawdown(df):
    df = ensure_return(df)
    wealth_index = 1000 * (1 + df["return"]).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    drawdown_df = pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdown
    })
    return drawdown_df

def plot_drawdown(df, coin):
    dd = compute_drawdown(df)
    dd[["Wealth", "Peaks"]].plot(figsize=(12, 6))
    dd["Drawdown"].plot(figsize=(12, 6), title=f"Drawdown for {coin}")
    plt.show()
    print(f"Max drawdown for {coin}: {dd['Drawdown'].min():.2%} at {dd['Drawdown'].idxmin()}")

# =======================
# 5. Return correlation heatmap
# =======================
def plot_return_correlation(dataframes):
    returns_dict = {}
    for name, df in dataframes.items():
        df = ensure_return(df)
        returns_dict[name] = df["return"]
    return_corr = pd.DataFrame(returns_dict).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(return_corr, annot=False, cmap="coolwarm")
    plt.title("Return Correlation Heatmap")
    plt.show()

# =======================
# 6. Volume correlation heatmap
# =======================
def plot_volume_correlation(dataframes):
    vols_dict = {}
    for name, df in dataframes.items():
        if "Volume" in df.columns:
            vols_dict[name] = df["Volume"]
    if vols_dict:
        vol_corr = pd.DataFrame(vols_dict).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(vol_corr, annot=False, cmap="coolwarm")
        plt.title("Volume Correlation Heatmap")
        plt.show()

# =======================
# 7. Volatility & Sharpe Ratio
# =======================
def compute_risk_metrics(dataframes):
    rows = []
    for name, df in dataframes.items():
        df = ensure_return(df)
        vol = df["return"].std()
        annual_vol = vol * np.sqrt(252)
        sharpe = (df["return"].mean() * 252 - 0.03) / annual_vol if annual_vol > 0 else np.nan
        rows.append({"Coin": name, "Volatility": annual_vol, "Sharpe": sharpe})
    metrics_df = pd.DataFrame(rows)
    return metrics_df.sort_values(by="Sharpe", ascending=False)

def plot_volatility_and_sharpe(metrics_df):
    plt.figure(figsize=(14, 6))
    plt.bar(metrics_df["Coin"], metrics_df["Volatility"])
    plt.title("Annual Volatility per Coin")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.bar(metrics_df["Coin"], metrics_df["Sharpe"])
    plt.title("Sharpe Ratio per Coin")
    plt.xticks(rotation=45)
    plt.show()
