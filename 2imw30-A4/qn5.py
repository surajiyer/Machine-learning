class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

from keras.datasets import mnist
from keras import backend as K

# (You don't need to change this part of the code)
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
y_train, y_test = y_train.astype('str'), y_test.astype('str')

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

from keras.engine.training import _slice_arrays

# Build training data
TRAINING_SIZE = 20000
DIGITS = 10
chars = '0123456789 '
ctable = CharacterTable(chars, DIGITS)
f = lambda: np.random.choice(X_train.shape[0], 
                             size=np.random.randint(1, DIGITS + 1))
X_new, y_new = [], []
seen = set()

while len(X_new) < TRAINING_SIZE:
    # Get indices of random digits for two numbers
    a, b = f(), f()
    
    # Get the images of the corresponding random digits
    x1, x2 = X_train[a], X_train[b]
    y1, y2 = int(''.join(y_train[a])), int(''.join(y_train[b]))
    data = (y1, y2)
    if data in seen:
        continue
    seen.add(data)
    
    # Pad the data with null images
    x1.resize((DIGITS, *x1.shape[1:]))
    x2.resize((DIGITS, *x1.shape[1:]))
    
    # Append the images data
    X_new.append([x1, x2])
    
    # Append the answer
    ans = str(y1 + y2)
    # Answers can be of maximum size DIGITS + 1
    ans += ' ' * (DIGITS + 1 - len(ans))
    y_new.append(ctable.encode(ans, maxlen=DIGITS+1))

# Save memory
del seen, x1, x2, data, y1, y2

X_new, y_new = np.array(X_new), np.array(y_new)

# Shuffle (X_new, y_new) in unison as the later parts of X_new 
# will almost all be larger digits
indices = np.arange(len(y_new))
np.random.shuffle(indices)
X_new = X_new[indices]
y_new = y_new[indices]

# Save memory!
del indices

# Explicitly set apart 10% for test data that we never train over
split_at = int(len(X_new) - len(X_new) / 10)
(X_train_new, X_test_new) = (_slice_arrays(X_new, 0, split_at), 
                             _slice_arrays(X_new, split_at))
(y_train_new, y_test_new) = (y_new[:split_at], y_new[split_at:])

# Save memory!!
del X_new, y_new

print(X_train_new.shape, y_train_new.shape)

# (You don't need to change this part of the code)
from __future__ import print_function
import numpy as np
np.random.seed(1234)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import recurrent, Conv2D, MaxPooling2D
from keras.layers import Reshape, Flatten, RepeatVector
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed

RNN = recurrent.SimpleRNN
HIDDEN_SIZE = 128
batch_size = 10

# Build the model
print('Build model...')
model = Sequential()

# Rehshape the input for the image identification layers
model.add(Reshape((-1, *X_train_new.shape[-3:]), 
                  input_shape=X_train_new.shape[1:],
                  batch_input_shape=(batch_size,)+X_train_new.shape[1:]))

# Image identification
model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3))))
model.add(Activation('relu'))
model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3))))
model.add(Activation('relu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

# Reshape the input for RNN
model.add(Reshape((*X_train_new.shape[1:3], -1)))
model.add(TimeDistributed(Flatten()))

# Learning to add numbers in images
model.add(RNN(HIDDEN_SIZE))
model.add(RepeatVector(DIGITS + 1))
model.add(RNN(HIDDEN_SIZE, stateful=False, return_sequences=True))

# For each of step of the output sequence, decide which 
# character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
[(x.input_shape, x.name, x.output_shape) for x in model.layers]

# Train the model
nr_epochs = 12
model.fit(X_train_new, y_train_new,
          batch_size=batch_size,
          epochs=nr_epochs,
          verbose=0,
          validation_data=(X_test_new, y_test_new))

# Evaluation
score = model.evaluate(X_test_new, y_test_new, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])