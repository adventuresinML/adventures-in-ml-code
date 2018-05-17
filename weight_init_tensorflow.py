import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial

base_path = "C:\\Users\\Andy\\PycharmProjects\\Tensorboard\\weights\\"

def maybe_create_folder_structure(sub_folders):
    for fold in sub_folders:
        if not os.path.isdir(base_path + fold):
            os.makedirs(base_path + fold)

class Model(object):
    def __init__(self, input_size, label_size, initialization, activation, num_layers=3,
                 hidden_size=100):
        self._input_size = input_size
        self._label_size = label_size
        self._init = initialization
        self._activation = activation
        # num layers does not include the input layer
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._model_def()

    def _model_def(self):
        # create placeholder variables
        self.input_images = tf.placeholder(tf.float32, shape=[None, self._input_size])
        self.labels = tf.placeholder(tf.float32, shape=[None, self._label_size])
        # create self._num_layers dense layers as the model
        input = self.input_images
        tf.summary.scalar("input_var", self._calculate_variance(input))
        for i in range(self._num_layers - 1):
            input = tf.layers.dense(input, self._hidden_size, kernel_initializer=self._init,
                                    activation=self._activation, name='layer{}'.format(i+1))
            # get the input to the nodes (sans bias)
            mat_mul_in = tf.get_default_graph().get_tensor_by_name("layer{}/MatMul:0".format(i + 1))
            # log pre and post activation function histograms
            tf.summary.histogram("mat_mul_hist_{}".format(i + 1), mat_mul_in)
            tf.summary.histogram("fc_out_{}".format(i + 1), input)
            # also log the variance of mat mul
            tf.summary.scalar("mat_mul_var_{}".format(i + 1), self._calculate_variance(mat_mul_in))
        # don't supply an activation for the final layer - the loss definition will
        # supply softmax activation. This defaults to a linear activation i.e. f(x) = x
        logits = tf.layers.dense(input, 10, name='layer{}'.format(self._num_layers))
        mat_mul_in = tf.get_default_graph().get_tensor_by_name("layer{}/MatMul:0".format(self._num_layers))
        tf.summary.histogram("mat_mul_hist_{}".format(self._num_layers), mat_mul_in)
        tf.summary.histogram("fc_out_{}".format(self._num_layers), input)
        # use softmax cross entropy with logits - no need to apply softmax activation to
        # logits
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                             labels=self.labels))
        # add the loss to the summary
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.accuracy = self._compute_accuracy(logits, self.labels)
        tf.summary.scalar('acc', self.accuracy)
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()

    def _compute_accuracy(self, logits, labels):
        prediction = tf.argmax(logits, 1)
        equality = tf.equal(prediction, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        return accuracy

    def _calculate_variance(self, x):
        mean = tf.reduce_mean(x)
        sqr = tf.square(x - mean)
        return tf.reduce_mean(sqr)

def init_pass_through(model, fold):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as sess:
        sess.run(model.init_op)
        train_writer = tf.summary.FileWriter(base_path + fold,
                                             sess.graph)
        image_batch, label_batch = mnist.train.next_batch(100)
        summary = sess.run(model.merged, feed_dict={model.input_images: image_batch,
                                                    model.labels: label_batch})
        train_writer.add_summary(summary, 0)

def train_model(model, fold, batch_size, epochs):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as sess:
        sess.run(model.init_op)
        train_writer = tf.summary.FileWriter(base_path + fold,
                                             sess.graph)
        for i in range(epochs):
            image_batch, label_batch = mnist.train.next_batch(batch_size)
            loss, _, acc = sess.run([model.loss, model.optimizer, model.accuracy],
                                    feed_dict={model.input_images: image_batch,
                                                    model.labels: label_batch})
            if i % 50 == 0:
                print("Iteration {} of {} - loss: {:.3f}, training accuracy: {:.2f}%".
                      format(i, epochs, loss, acc*100))
                summary = sess.run(model.merged, feed_dict={model.input_images: image_batch,
                                                    model.labels: label_batch})
                train_writer.add_summary(summary, i)


if __name__ == "__main__":
    sub_folders = ['first_pass_normal', 'first_pass_variance',
                   'full_train_normal', 'full_train_variance',
                   'full_train_normal_relu', 'full_train_variance_relu',
                   'full_train_he_relu']
    initializers = [tf.random_normal_initializer,
                    tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False),
                    tf.random_normal_initializer,
                    tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False),
                    tf.random_normal_initializer,
                    tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False),
                    tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)]
    activations = [tf.sigmoid, tf.sigmoid, tf.sigmoid, tf.sigmoid, tf.nn.relu, tf.nn.relu, tf.nn.relu]
    assert len(sub_folders) == len(initializers) == len(activations)
    maybe_create_folder_structure(sub_folders)
    for i in range(len(sub_folders)):
        tf.reset_default_graph()
        model = Model(784, 10, initializers[i], activations[i])
        if "first_pass" in sub_folders[i]:
            init_pass_through(model, sub_folders[i])
        else:
            train_model(model, sub_folders[i], 30, 1000)

