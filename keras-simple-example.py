# Simple example
# Based on MNIST-MLP_from_raw_data

# Loading the packages

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils

# Load data & split data between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

# Define hyperparameters
n_classes = 10
E = 5                # number of epochs
B = 128              # batch size
D = X_train.shape[1] # dimension of input sample - 784 for MNIST

# Conversion to class vectors
Y_train = utils.to_categorical(y_train, n_classes)
Y_test = utils.to_categorical(y_test, n_classes)

# one layer network
model = Sequential()
model.add(Dense(n_classes, input_shape=(D,), activation='relu'))
model.summary()

# Compile and train the network
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])
log = model.fit(X_train, Y_train, batch_size=B, epochs=E,
                    verbose=1, shuffle=True, validation_data=(X_test, Y_test))

# Model evaluation
loss_test, metric_test = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss_test)
print('Test accuracy:', metric_test)

