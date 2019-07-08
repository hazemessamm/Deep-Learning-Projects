import numpy as np
from keras.datasets import reuters

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old


word_index = reuters.get_word_index()

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

decoded_review = ''.join(reverse_word_index.get(i-3, '?') for i in train_data[0])

def vectorize_sequence(sequence, dimension = 10000):
    results = np.zeros((len(sequence), dimension))
    for i, seq in enumerate(sequence):
        results[i, seq] = 1
    return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

from keras.utils.np_utils import to_categorical
import tensorflow as tf
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

from keras.layers import Dense
from keras.models import Sequential

#I am using Tanh just for further experiments but RELU is recommended.
model = Sequential()
model.add(Dense(64, activation=tf.nn.tanh, input_shape=(10000,)))
model.add(Dense(64, activation=tf.nn.tanh))
model.add(Dense(46, activation=tf.nn.softmax))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

val_x = x_train[0:1000]
partial_x_train = x_train[1000:]

val_y = y_train[0:1000]
partial_y_train = y_train[1000:]

model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=128, validation_data=(val_x, val_y))
