import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

def prepare_data(csv_path, seq_length=4, test_size=0.2, random_state=42):
    """
    Load data from csv_path, unify column names, sort by delay.
    Normalize delay, arousal, valence, stimulation using MinMaxScaler.
    Create sliding window sequences.
    Split into train/validation sets and return tensors along with scalers.
    """
    df = pd.read_csv(csv_path)
    df.columns = ['delay', 'arousal', 'valence', 'stimulation', 'sample']
    df.sort_values('delay', inplace=True)
    df.reset_index(drop=True, inplace=True)

    features = df[['delay', 'arousal', 'valence', 'stimulation']].values.astype(np.float32)
    targets = df[['arousal', 'valence', 'stimulation']].values.astype(np.float32)

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    features_norm = scaler_X.fit_transform(features)
    targets_norm = scaler_Y.fit_transform(targets)

    X_seq, Y_seq = [], []
    for i in range(len(features_norm) - seq_length):
        X_seq.append(features_norm[i: i + seq_length])
        Y_seq.append(targets_norm[i + seq_length])

    X_seq = np.stack(X_seq)  # (num_samples, seq_length, 4)
    Y_seq = np.stack(Y_seq)  # (num_samples, 3)

    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
        X_seq, Y_seq, test_size=test_size, random_state=random_state, shuffle=True
    )

    X_train = torch.from_numpy(X_train_np)
    Y_train = torch.from_numpy(Y_train_np)
    X_val = torch.from_numpy(X_val_np)
    Y_val = torch.from_numpy(Y_val_np)

    return X_train, Y_train, X_val, Y_val, scaler_Y, scaler_X