import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(filepath, save_path):
    """
    Preprocess Bitcoin price data for RNN training.
    
    Args:
        filepath (str): Path to the raw dataset.
        save_path (str): Path to save preprocessed data.
    """
    # Load dataset
    data = pd.read_csv(filepath)

    # Use only the close price
    data = data[['close']]

    # Normalize the data
    scaler = MinMaxScaler()
    data['close'] = scaler.fit_transform(data[['close']])

    # Create sequences of 24 hours (1440 minutes)
    sequence_length = 1440
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length - 60):
        seq = data['close'].iloc[i:i + sequence_length].values
        target = data['close'].iloc[i + sequence_length + 59]
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42)

    # Save preprocessed data
    np.savez(save_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, scaler=scaler)

if __name__ == "__main__":
    preprocess_data("coinbase.csv", "preprocessed_data.npz")
