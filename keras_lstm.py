'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import one_hot
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import argparse

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "C:\\Users\Andy\Documents\simple-examples\data"

parser = argparse.ArgumentParser()
# parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('data_path', type=str, default=data_path, help='The full path of the training data')
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

    def __init__(self, data, num_steps, target_size, batch_size, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.target_size = target_size
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((batch_size, self.num_steps))
        y = np.zeros((batch_size, self.target_size))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps + self.target_size >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                y[i, :] = self.data[self.current_idx + self.num_steps:self.current_idx + self.num_steps + self.target_size]
                self.current_idx += self.skip_step
            yield x, y

num_steps = 25
target_size = 10
batch_size = 100
train_data_generator = KerasBatchGenerator(train_data, num_steps, target_size, batch_size)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, target_size, batch_size)

hidden_layers = 300

model = Sequential()
model.add(Embedding(vocabulary, hidden_layers, input_length=num_steps))
model.add(LSTM(hidden_layers, return_sequences=True))
model.add(LSTM(hidden_layers))
# model.add(Flatten())
model.add(Dense(target_size))
model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

num_epochs = 20
# model.fit_generator(train_data_generator.generate(), len(train_data)//batch_size, num_epochs,
#                     validation_data=valid_data_generator.generate(),
#                     validation_steps=len(valid_data)//batch_size)

model.fit_generator(train_data_generator.generate(), len(train_data)//batch_size, num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//batch_size)
