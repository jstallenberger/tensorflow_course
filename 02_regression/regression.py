import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/project/02_regression/moore.csv', header=None).values
X = data[:,0].reshape(-1, 1) # make it a 2-D array of size N x D where D = 1
Y = data[:,1]

# plt.scatter(X, Y)
Y = np.log(Y) # taking the log of Y because of exponential data
# plt.scatter(X, Y)

# X data is years (from 1970 to 2020)
# Let's center the X data around 0 so the values are not too large
# We could scale it but then we would have to revers the transformation later
X = X - X.mean()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse') # SGD: learning rate, momentum
# model.compile(optimizer='adam', loss='mse')

# Learning Rate Scheduler
def schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

plt.plot(r.history['loss'], label='loss')
plt.savefig('/project/02_regression/loss.png')

# Get the slope of the line
# The slope of the line is related to the doubling rate of transistor count
print(model.layers)
print(model.layers[0].get_weights()) # Note: there is only one layer, the input layer does not count

# The slope of the line is:
a = model.layers[0].get_weights()[0][0,0]

print("Time to double:", np.log(2) / a)

# Making predictions

# Make sure the line fits our data
Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.savefig('/project/02_regression/predicted_line.png')

# Manual calculation

# Get the weights
w, b = model.layers[0].get_weights()

# Reshape X because we flattened it again earlier
X = X.reshape(-1, 1)

# (N x 1) x (1 x 1) + (1) --> (N x 1)
Yhat2 = (X.dot(w) + b).flatten()

# Don't use == for floating points
np.allclose(Yhat, Yhat2)