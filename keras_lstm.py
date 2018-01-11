from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
import argparse
import pdb

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "C:\\Users\Andy\Documents\simple-examples\data"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, target_size, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.target_size = target_size
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.target_size, self.vocabulary))
        temp_y = np.zeros((self.batch_size, self.target_size))
        # temp_y_2 = np.zeros((self.batch_size, self.target_size, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps + self.target_size >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y[i, :] = self.data[self.current_idx + self.num_steps:self.current_idx
                                                                           + self.num_steps + self.target_size]
                # convert all of temp_y into a one hot representation
                # temp_y_2 = np.zeros((self.target_size, self.vocabulary))
                # for j in range(self.target_size):
                 #   temp_y_2[i, j, temp_y[i, j]] = 1
                # now reshape to feed to softmax output layer, size = target_size * vocabulary
                # y[i, :] = np.reshape(temp_y_2, (1, self.target_size * self.vocabulary))
                y[i, :, :] = to_categorical(temp_y[i, :], num_classes=self.vocabulary)
                # pdb.set_trace()
                self.current_idx += self.skip_step
            yield x, y

num_steps = 25
target_size = 10
batch_size = 20
train_data_generator = KerasBatchGenerator(train_data, num_steps, target_size, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, target_size, batch_size, vocabulary,
                                           skip_step=num_steps)
test_data_generator = KerasBatchGenerator(test_data, num_steps, target_size, batch_size, vocabulary,
                                          skip_step=num_steps)

hidden_layers = 300

model = Sequential()
model.add(Embedding(vocabulary, hidden_layers, input_length=num_steps))
model.add(LSTM(hidden_layers, return_sequences=True))
# model.add(LSTM(hidden_layers, return_sequences=True))
# model.add(Dropout(0.2))
model.add(Reshape((target_size, -1)))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))
# model.add(Reshape((target_size*vocabulary,)))

def our_accuracy(y_true, y_pred, target_size=10):
    y_true = K.reshape(y_true, (target_size, -1))
    y_pred = K.reshape(y_pred, (target_size, -1))
    # pdb.set_trace()
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

def loss(y_true, y_pred, target_size=10):
    y_true = K.reshape(y_true, (target_size, -1))
    y_pred = K.reshape(y_pred, (target_size, -1))


# optimizer = RMSprop(lr=0.001)
optimizer = SGD(lr=1.0)
# optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

print(model.summary())

num_epochs = 50
if args.run_opt == 1:
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps))
    # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
    #                     validation_data=valid_data_generator.generate(),
    #                     validation_steps=10)
    model.save(data_path + "model.h5")
elif args.run_opt == 2:
    model = load_model(data_path + "model.h5")
    prediction = model.predict_generator(train_data_generator.generate(),steps=1)
    pdb.set_trace()
    x=1





