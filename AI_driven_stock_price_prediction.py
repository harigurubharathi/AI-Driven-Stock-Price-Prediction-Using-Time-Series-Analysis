import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the stock data (replace 'your_stock_data.csv' with your actual file)
df = pd.read_csv('your_stock_data.csv', index_col='Date', parse_dates=True)

# Prepare the data
# Select the 'Close' price for simplicity, you can use other features
mid_prices = (df['High'] + df['Low']) / 2
train_data = mid_prices[:-30]  # Use all but last 30 days for training
test_data = mid_prices[-30:]   # Last 30 days for testing

# Scale the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
test_data = scaler.transform(test_data.values.reshape(-1, 1)).flatten()

# Create sequences for LSTM
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(x), np.array(y)

seq_length = 30
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Reshape for LSTM input (samples, time steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(x_test)

# Inverse transform the predictions and actual values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model (Mean Squared Error)
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual', color='blue')
plt.plot(y_pred_inv, label='Predicted', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()