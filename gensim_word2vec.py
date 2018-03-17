import gensim
from gensim.models import word2vec
import logging

from keras.layers import Input, Embedding, merge
from keras.models import Model

import tensorflow as tf
import numpy as np

import urllib.request
import os
import zipfile

vector_dim = 300
root_path = "C:\\Users\Andy\PycharmProjects\\adventures-in-ml-code\\"

def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

# convert the input data into a list of integer indexes aligning with the wv indexes
# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

def gensim_demo():
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    if not os.path.exists((root_path + filename).strip('.zip')):
        zipfile.ZipFile(root_path+filename).extractall()
    sentences = word2vec.Text8Corpus((root_path + filename).strip('.zip'))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, iter=10, min_count=10, size=300, workers=4)

    # get the word vector of "the"
    print(model.wv['the'])

    # get the most common words
    print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])

    # get the least common words
    vocab_size = len(model.wv.vocab)
    print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])

    # find the index of the 2nd most common word ("of")
    print('Index of "of" is: {}'.format(model.wv.vocab['of'].index))

    # some similarity fun
    print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'elephant'))

    # what doesn't fit?
    print(model.wv.doesnt_match("green blue red zebra".split()))

    str_data = read_data(root_path + filename)
    index_data = convert_data_to_index(str_data, model.wv)
    print(str_data[:4], index_data[:4])

    # save and reload the model
    model.save(root_path + "mymodel")


def create_embedding_matrix(model):
    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # into our TensorFlow and Keras models
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def tf_model(embedding_matrix, wv):
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # embedding layer weights are frozen to avoid updating embeddings while training
    saved_embeddings = tf.constant(embedding_matrix)
    embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

    # create the cosine similarity operations
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = embedding / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # call our similarity operation
        sim = similarity.eval()
        # run through each valid example, finding closest words
        for i in range(valid_size):
            valid_word = wv.index2word[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = wv.index2word[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)


def keras_model(embedding_matrix, wv):
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    # input words - in this case we do sample by sample evaluations of the similarity
    valid_word = Input((1,), dtype='int32')
    other_word = Input((1,), dtype='int32')
    # setup the embedding layer
    embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix])
    embedded_a = embeddings(valid_word)
    embedded_b = embeddings(other_word)
    similarity = merge([embedded_a, embedded_b], mode='cos', dot_axes=2)
    # create the Keras model
    k_model = Model(input=[valid_word, other_word], output=similarity)

    def get_sim(valid_word_idx, vocab_size):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = k_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

    # now run the model and get the closest words to the valid examples
    for i in range(valid_size):
        valid_word = wv.index2word[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        sim = get_sim(valid_examples[i], len(wv.vocab))
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = wv.index2word[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        gensim_demo()
    elif run_opt == 2:
        model = gensim.models.Word2Vec.load(root_path + "mymodel")
        embedding_matrix = create_embedding_matrix(model)
        tf_model(embedding_matrix, model.wv)
    elif run_opt == 3:
        model = gensim.models.Word2Vec.load(root_path + "mymodel")
        embedding_matrix = create_embedding_matrix(model)
        keras_model(embedding_matrix, model.wv)
