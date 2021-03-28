from tensorflow.keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load in the data
df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')

print(df.head())
print(df.tail())

# Calculate returns by first shifting the data
df['PrevClose'] = df['close'].shift(1) # move everything up 1

# so now it's like:
# close / prev close
# x[2] x[1]
# x[3] x[2]
# x[4] x[3]
# ...
# x[t] x[t-1]

print(df.head())

# then the return is:
# (x[t] - x[t-1]) / x[t-1]
df['Return'] = (df['close'] - df['PrevClose']) / df['PrevClose']

print(df.head())

# Now let's try an LSTM to predict returns
df['Return'].hist()
plt.show()
plt.savefig('/project/14_LSTM_stock_returns/2_stock_returns_histogram.png')
plt.close()

series = df['Return'].values[1:].reshape(-1, 1)

# Normalize the data
# Note: the true boundary is just an approximation
scaler = StandardScaler()
scaler.fit(series[:len(series) // 2]) # calling fit function on the first half of the series only, to not include test data in the training pipeline
series = scaler.transform(series).flatten() # transform is called on the entire dataset, then flatten to get an N-length vector

# Build the dataset
# Let's see if we can use T past values to predict the next value
T = 10
D = 1
X = []
Y = []
for t in range(len(series) - T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, 1) # Now the data should be N x T x D
Y = np.array(Y)
N = len(X)
print("X.shape: ", X.shape, ", Y.shape: ", Y.shape)

# Try an LSTM network
i = Input(shape=(T, 1))
x = LSTM(5)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.1)
)

# Train the LSTM
r = model.fit(
    X[:-N//2], Y[:-N//2],
    epochs=80,
    validation_data=(X[-N//2:], Y[-N//2:])
)

# Plot loss
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig('/project/14_LSTM_stock_returns/2_stock_return_loss.png')
plt.close()

# One-step forecast using true targets
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]

plt.plot(Y, label='targets')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()
plt.savefig('/project/14_LSTM_stock_returns/2_stock_return_onestep_predictions.png')
plt.close()

# Multi-step forecast
validation_target = Y[-N//2:]
validation_predictions = []

# Last train input
last_x = X[-N//2] # 1-D array of length T

while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, T, 1))[0,0] # 1x1 array -> scalar

    # update the predictions list
    validation_predictions.append(p)

    # make the new input
    last_x = np.roll(last_x, -1)
    last_x[-1] = p

plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()
plt.savefig('/project/14_LSTM_stock_returns/2_stock_return_multistep_predictions.png')
plt.close()