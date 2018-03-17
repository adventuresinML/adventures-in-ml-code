import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits

def simple_dataset_with_error():
    x = np.arange(0, 10)
    # create dataset object from the numpy array
    dx = tf.data.Dataset.from_tensor_slices(x)
    # create a one-shot iterator
    iterator = dx.make_one_shot_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(11):
            val = sess.run(next_element)
            print(val)

def simple_dataset_initializer():
    x = np.arange(0, 10)
    dx = tf.data.Dataset.from_tensor_slices(x)
    # create an initializable iterator
    iterator = dx.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if i % 9 == 0 and i > 0:
                sess.run(iterator.initializer)

def simple_dataset_batch():
    x = np.arange(0, 10)
    dx = tf.data.Dataset.from_tensor_slices(x).batch(3)
    # create a one-shot iterator
    iterator = dx.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)

def simple_zip_example():
    x = np.arange(0, 10)
    y = np.arange(1, 11)
    # create dataset objects from the arrays
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)
    # zip the two datasets together
    dcomb = tf.data.Dataset.zip((dx, dy)).batch(3)
    iterator = dcomb.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)

def MNIST_dataset_example():
    # load the data
    digits = load_digits(return_X_y=True)
    # split into train and validation sets
    train_images = digits[0][:int(len(digits[0]) * 0.8)]
    train_labels = digits[1][:int(len(digits[0]) * 0.8)]
    valid_images = digits[0][int(len(digits[0]) * 0.8):]
    valid_labels = digits[1][int(len(digits[0]) * 0.8):]
    # create the training datasets
    dx_train = tf.data.Dataset.from_tensor_slices(train_images)
    # apply a one-hot transformation to each label for use in the neural network
    dy_train = tf.data.Dataset.from_tensor_slices(train_labels).map(lambda z: tf.one_hot(z, 10))
    # zip the x and y training data together and shuffle, batch etc.
    train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)
    # do the same operations for the validation set
    dx_valid = tf.data.Dataset.from_tensor_slices(valid_images)
    dy_valid = tf.data.Dataset.from_tensor_slices(valid_labels).map(lambda z: tf.one_hot(z, 10))
    valid_dataset = tf.data.Dataset.zip((dx_valid, dy_valid)).shuffle(500).repeat().batch(30)
    # create general iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    next_element = iterator.get_next()
    # make datasets that we can initialize separately, but using the same structure via the common iterator
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(valid_dataset)
    # create the neural network model
    logits = nn_model(next_element[0])
    # add the optimizer and loss
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # get accuracy
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    init_op = tf.global_variables_initializer()
    # run the training
    epochs = 600
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(training_init_op)
        for i in range(epochs):
            l, _, acc = sess.run([loss, optimizer, accuracy])
            if i % 50 == 0:
                print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
        # now setup the validation run
        valid_iters = 100
        # re-initialize the iterator, but this time with validation data
        sess.run(validation_init_op)
        avg_acc = 0
        for i in range(valid_iters):
            acc = sess.run([accuracy])
            avg_acc += acc[0]
        print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,
                                                                                     (avg_acc / valid_iters) * 100))

def nn_model(in_data):
    bn = tf.layers.batch_normalization(in_data)
    fc1 = tf.layers.dense(bn, 50)
    fc2 = tf.layers.dense(fc1, 50)
    fc2 = tf.layers.dropout(fc2)
    fc3 = tf.layers.dense(fc2, 10)
    return fc3


if __name__ == "__main__":
    # simple_dataset_with_error()
    # simple_dataset_initializer()
    # simple_dataset_batch()
    # simple_zip_example()
    MNIST_dataset_example()