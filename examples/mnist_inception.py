from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from six.moves import zip

class Inception(Layer):
    '''
        Inception Layer from GoogLeNet.
    '''
    def __init__(self, inputDepth, n1x1conv, n3x3reduce, n3x3conv):
        self.l1x1conv = Convolution2D(n1x1conv, inputDepth, 1, 1, activation = 'relu')
        self.l3x3reduce = Convolution2D(n3x3reduce, inputDepth, 1, 1, activation = 'relu')
        self.l3x3conv = Convolution2D(n3x3conv, n3x3reduce, 3, 3, activation = 'relu')
        #self.params = self.l1x1conv.params + self.l3x3reduce.params + self.l3x3conv.params   
        self.params = self.l1x1conv.params
    
    def get_output(self, train=False):
        X = self.get_input(train)
        print(X)
        
        # 1x1 convolution
        self.l1x1conv.input = X
        out1x1 = self.l1x1conv.get_output(train)
        return out1x1
       ''' 
        # 3x3 convolution
        self.l3x3reduce.input = X
        self.l3x3conv.input = self.l3x3reduce.get_output(train)
        out3x3 = self.l3x3conv.get_output(train)
        
        return T.concatenate([out1x1, out3x3], axes = 1)
        '''

    def set_name(self, name):
        self.l1x1conv.set_name('%s_1x1conv' % name)
        self.l3x3reduce.set_name('%s_3x3reduce' % name)
        self.l3x3conv.set_name('%s_3x3conv' % name)
        self.b.name = '%s_b' % name

    def get_config(self):
        return {
                "name": self.__class__.__name__,
                "l1x1conv": self.l1x1conv.get_config(),
                "l3x3reduce": self.l3x3reduce.get_config(),
                "l3x3conv": self.l3x3conv.get_config(),
        }

'''
    Train a simple convnet on the MNIST dataset.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 12

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
#incep = Inception(1, 128, 64, 64)
#print(incep.get_config())
#model.add(incep)
model.add(Convolution2D(32, 1, 3, 3))
model.add(Dense(128, nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
