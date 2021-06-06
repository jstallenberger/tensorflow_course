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

# Freeze pretrained model weights
ptm.trainable = False

# Create the head of the neural network

# Map the data into feature vectors
# Keras image data generator returns classes one-hot encoded

K = len(folders) # number of classes
x = Flatten()(ptm.output)
x = Dense(K, activation='softmax')(x) # softmax is a natural choice where the targets are one-hot encoded
                                      # softmax also works for more classes, not just 2

# Create a model object
model = Model(inputs=ptm.input, outputs=x)

# View the summary of the model
print(model.summary())

# Create an instance of ImageDataGenerator
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

batch_size = 128

# Create generators
train_generator = gen.flow_from_directory(
    train_path,
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=batch_size
)

valid_generator = gen.flow_from_directory(
    valid_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size
)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Fit the model
r = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    steps_per_epoch=int(np.ceil(len(image_files) / batch_size)),
    validation_steps=int(np.ceil(len(valid_image_files) / batch_size)),
)

# Loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('/project/17_transfer_learning/loss_augmented.png')
plt.close()

# Accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('/project/17_transfer_learning/accuracy_augmented.png')
plt.close()
