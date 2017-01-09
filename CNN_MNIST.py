
# import the built-in function to get MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D


# import numpy and seed for reproducibility
import numpy as np
np.random.seed(64)
# downloand the MNIST dataset and split into (x,y_) for training and (x_test,y_test) for testing.
# x is pixel values of each images and y_ is corresponding labels.

(x,y_),(x_test,y__) = mnist.load_data()

#%% Data preprocess
# Step 1: reshape the image to the format [sample #][width][height][channel] for Theano backend
from keras import backend as K
K.set_image_dim_ordering('th')

x = x.reshape(x.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

# Step 2: Normalize the pixel gray-scale values
x = x/255
x_test = x_test/255
# Step 3: Explode the label values to binary vector. To do this we need an utility. 
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_)
y_test = np_utils.to_categorical(y__)
print(x.shape[0], 'training exmples')
print(x_test.shape[0], 'test samples')

#%%
# batch size
batch = 128
# number of epoches
epoch = 25
# number of convolutional filters
filters = 32
# convolution filter size
filter_size = (3, 3)
# pooling size
pooling_size = (2, 2)
#%%

# CNN is a sequential model
model = Sequential()

# Layer struture:

# Conv 1: First convolutional layer. Arguments: number of filters, filter size, padding mode, input structure.
model.add(Convolution2D(filters, filter_size[0], filter_size[1],
                        border_mode='valid',input_shape=(1,28,28)))
# Activ1: First activation layer using relu activation function.
model.add(Activation('relu'))
# Conv 2: Second convolutional layer.
model.add(Convolution2D(filters, filter_size[0], filter_size[1]))
# Activ2: Second activation layer using relu activation function.
model.add(Activation('relu'))
# MaxPool1: First pooling layer using maxpooling method, Argument: pooling size.
model.add(MaxPooling2D(pool_size=pooling_size))
# Dropout1: First dropout layer. Drop (set to zero) 25% of neurons.
model.add(Dropout(0.25))
# Flatten1: First flatten layer.
model.add(Flatten())
# FC1: First fully connected layer.
model.add(Dense(128))
# Activ 3: Third activation layer using relu activation function.
model.add(Activation('relu'))
# Dropout2: Second dropout layer. Drop (set to zero) 50% of neurons.
model.add(Dropout(0.5))
# FC2: Second fully connected layer.
model.add(Dense(10))
# Activ 3: Forth activation layer using softmax activation function.
model.add(Activation('softmax'))
#%%
# compile model by specifying loss function, opimizer, and evaluation metric. In fact, optimizer could 
# a hyperparemet to experiment with
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(x, y_train, batch_size=batch, nb_epoch=epoch,verbose=1, 
          validation_data=(x_test, y_test))
result = model.evaluate(x_test, y_test, verbose=0)
print('Test result is {} and test accuracy is {}'.format(result[0],result[1]))


