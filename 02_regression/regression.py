import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('moore.csv', header=None).values
X = data[:,0].reshape(-1, 1) # make it a 2-D array of size N x D where D = 1
Y = data[:,1]

# plt.scatter(X, Y)
Y = np.log(Y)
# plt.scatter(X, Y)

X = X - X.mean()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse') # SGD: learning rate, momentum
# model.compile(optimizer='adam', loss='mse'

# Learning Rate Scheduler
def schedule(epoch, lr):
  if epoch >= 50:
    return 0.0001
  return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

plt.plot(r.history['loss'], label='loss')
plt.savefig('loss.png')

# The slope of the line is:
a = model.layers[0].get_weights()[0][0,0]

print("Time to double:", np.log(2) / a)
