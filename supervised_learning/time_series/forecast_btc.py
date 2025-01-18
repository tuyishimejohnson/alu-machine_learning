import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np

def build_model(input_shape):
    """
    Builds an RNN model for forecasting BTC prices.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features).
    
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_evaluate(data_path, model_path):
    """
    Trains and evaluates the RNN model for BTC forecasting.
    
    Args:
        data_path (str): Path to preprocessed data.
        model_path (str): Path to save the trained model.
    """
    # Load preprocessed data
    data = np.load(data_path)
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

    # Build the model
    model = build_model((X_train.shape[1], 1))

    # Train the model
    model.fit(train_dataset, epochs=10, validation_data=test_dataset)

    # Save the model
    model.save(model_path)

    # Evaluate the model
    test_loss = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    train_and_evaluate("preprocessed_data.npz", "btc_forecast_model.h5")
