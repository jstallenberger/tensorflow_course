from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Build the dataset
# This is a nonlinear AND long-distance dataset
# We will test long-distance vs. short-distance patterns

# Start with a small T (e.g. 10) and increase it later
T = 30
D = 1
X = []
Y = []

def get_label(x, i1, i2, i3):
    # x = sequence
    if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
        return 1
    if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
        return 1
    return 0

for t in range(5000):
    x = np.random.randn(T)
    X.append(x)
    # y = get_label(x, -1, -2, -3) # short distance
    y = get_label(x, 0, 1, 2) # long distance
    Y.append(y)

X = np.array(X)
Y = np.array(Y)
N = len(X)

# Test the LSTM
inputs = np.expand_dims(X, -1)

i = Input(shape=(T, D))

x = LSTM(5, return_sequences=True)(i) # return_sequences=True: gets the hidden states for each timestep -> needs pooling afterwards (T x M instead of M)
# For images: GlobalMaxPooling2D, for sequences: GlobalMaxPooling1D (element-wise max() operation)
x = GlobalMaxPooling1D()(x)

x = Dense(1, activation='sigmoid')(x)
model = Model(i, x)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)

r = model.fit(
    inputs, Y,
    epochs=100,
    validation_split=0.5
)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig('/project/12_LSTM_longdistance/loss.png')
plt.close()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig('/project/12_LSTM_longdistance/accuracy.png')
plt.close()
