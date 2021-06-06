from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# Food-5K.zip
# Data from: https://mmspg.epfl.ch/downloads/food-image-datasets
# Alternatively: https://www.kaggle.com/binhminhs10/food5k

# Food images start with 1, non-food images start with 0

# Make directories to store the data Keras-style

# mkdir data
# mkdir data/train
# mkdir data/test
# mkdir data/train/nonfood
# mkdir data/train/food
# mkdir data/test/nonfood
# mkdir data/test/food

# Move the images
# Note: consider 'training' to be the train set
#       'validation' to be the test set
#       ignore the 'evaluation' set

# mv training/0*.jpg data/train/nonfood
# mv training/1*.jpg data/train/food
# mv validation/0*.jpg data/test/nonfood
# mv validation/1*.jpg data/test/food

train_path = '/project/17_transfer_learning/data/train'
valid_path = '/project/17_transfer_learning/data/test'

# These images are pretty big and different sizes
# Let's load them all in as the same (smaller) size
IMAGE_SIZE = [200, 200]

# Useful for getting number of files
image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')

# Useful for getting number of classes
folders = glob(train_path + '/*')
print(folders)


ptm = PretrainedModel(
    input_shape=IMAGE_SIZE + [3], # +3 for the number of color channels
    weights='imagenet', # downloads weights trained on the ImageNet dataset
    include_top=False # instead of the full pretrained network, we only get the layers up to the final convolution, no flatten and final dense layers
)


# Map the data into feature vectors
x = Flatten()(ptm.output)

# Create a model object
model = Model(inputs=ptm.input, outputs=x)

# View the summary of the model
print(model.summary())

# Create an instance of ImageDataGenerator
gen = ImageDataGenerator(preprocessing_function=preprocess_input)

batch_size = 128

# Create generators
train_generator = gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='binary'
)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode='binary'
)

Ntrain = len(image_files)
Nvalid = len(valid_image_files)

# Figure out the output size
feat = model.predict(np.random.random([1] + IMAGE_SIZE + [3]))
D = feat.shape[1]

X_train = np.zeros((Ntrain, D))
Y_train = np.zeros(Ntrain)
X_valid = np.zeros((Nvalid, D))
Y_valid = np.zeros(Nvalid)

# Populate X_train and Y_train
i = 0
for x, y in train_generator:
    # get feature
    features = model.predict(x)

    # size of the batch (may not always be batch_size)
    sz = len(y)

    # assign to X_train and Y_train
    X_train[i:i + sz] = features
    Y_train[i:i + sz] = y

    # increment i
    i += sz
    print(i)
    if i >= Ntrain:
        print('breaking now')
        break
print(i)

# Populate X_valid and Y_valid
i = 0
for x, y in valid_generator:
    # get feature
    features = model.predict(x)

    # size of the batch (may not always be batch_size)
    sz = len(y)

    # assign to X_train and Y_train
    X_valid[i:i + sz] = features
    Y_valid[i:i + sz] = y

    # increment i
    i += sz
    print(i)
    if i >= Nvalid:
        print('breaking now')
        break
print(i)

print(X_train.max(), X_train.min())

# Max is too high so normalize the data with StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train2 = scaler.fit_transform(X_train)
X_valid2 = scaler.fit_transform(X_valid)

# Try the built-in logistic regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train2, Y_train)
print(logr.score(X_train2, Y_train))
print(logr.score(X_valid2, Y_valid))

# Do logistic regression in Tensorflow
i = Input(shape=(D,))
x = Dense(1, activation='sigmoid')(i)
linearmodel = Model(i, x)

linearmodel.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Can try both normalized and unnormalized data
r = linearmodel.fit(
    X_train, Y_train,
    batch_size = 128,
    epochs=10,
    validation_data=(X_valid, Y_valid)
)

# Loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('/project/17_transfer_learning/loss.png')
plt.close()

# Accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('/project/17_transfer_learning/accuracy.png')
plt.close()
