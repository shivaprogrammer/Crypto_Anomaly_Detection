import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_all_coins(data_path):
    """
    Load all CSV files in the given directory into a dictionary of DataFrames.
    """
    dataframes = {}
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_path, file))
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values("Date").set_index("Date")
            name = "coin_" + file.replace(".csv", "")
            dataframes[name] = df
    return dataframes

def prepare_features(df, test_size=0.2, scaler_type="standard"):
    """
    Preprocess a single coin dataframe:
    - Compute returns
    - Drop NaNs
    - Scale features using only the training split
    - Allow choice of StandardScaler or MinMaxScaler
    """

    #  Compute returns
    df['return'] = df['Close'].pct_change()
    df = df.dropna()

    #  Select useful features
    features = df[['Open', 'High', 'Low', 'Close', 'Volume', 'return']].values

    # Train/val split 
    n_total = features.shape[0]
    n_train = int((1 - test_size) * n_total)

    train_features = features[:n_train]
    test_features = features[n_train:]

    #  Choose scaler
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError("Invalid scaler_type. Choose 'standard' or 'minmax'.")

    #  Fit only on training set
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    #  Merge back
    features_scaled = np.vstack([train_scaled, test_scaled])
    df_scaled = pd.DataFrame(
        features_scaled,
        index=df.index,
        columns=['Open', 'High', 'Low', 'Close', 'Volume', 'return']
    )

    return df_scaled, scaler
