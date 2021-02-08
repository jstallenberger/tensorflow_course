import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load in the data
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten() # need to flatten as targets ar 2D arrays, Nx1 instead of N

# Load the model
model = tf.keras.models.load_model("/project/07_CNN_CIFAR-10_improved/model.h5")

p_test = model.predict(x_test).argmax(axis=1)

# Label mapping
labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()

# Show some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]))
plt.savefig("/project/07_CNN_CIFAR-10_improved/random_misclassified_sample.png")
plt.close()
