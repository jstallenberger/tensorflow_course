import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load in the data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Load the model
model = tf.keras.models.load_model("/project/03_ANN_classification_MNIST/model.h5")

# Make predictions
p_test = model.predict(x_test).argmax(axis=1)

# Show random prediction
i = np.random.randint(10000)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s, Predicted label: %s" % (y_test[i], p_test[i]))
plt.savefig("/project/03_ANN_classification_MNIST/random_sample.png")
plt.close()

# Show random misclassified prediction
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s, Predicted label: %s" % (y_test[i], p_test[i]))
plt.savefig("/project/03_ANN_classification_MNIST/random_misclassified_sample.png")
plt.close()
