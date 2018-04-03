from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

base_path = "C:\\Users\\Andy\\PycharmProjects\\Tensorboard\\"

class Model(object):
    def __init__(self, input_size, label_size, activation, num_layers=6,
                 hidden_size=10):
        self._input_size = input_size
        self._label_size = label_size
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
        for i in range(self._num_layers - 1):
            input = tf.layers.dense(input, self._hidden_size, activation=self._activation,
                                    name='layer{}'.format(i+1))
        # don't supply an activation for the final layer - the loss definition will
        # supply softmax activation. This defaults to a linear activation i.e. f(x) = x
        logits = tf.layers.dense(input, 10, name='layer{}'.format(self._num_layers))
        # use softmax cross entropy with logits - no need to apply softmax activation to
        # logits
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                             labels=self.labels))
        # add the loss to the summary
        tf.summary.scalar('loss', self.loss)
        self._log_gradients(self._num_layers)
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

    def _log_gradients(self, num_layers):
        gr = tf.get_default_graph()
        for i in range(num_layers):
            weight = gr.get_tensor_by_name('layer{}/kernel:0'.format(i + 1))
            grad = tf.gradients(self.loss, weight)[0]
            mean = tf.reduce_mean(tf.abs(grad))
            tf.summary.scalar('mean_{}'.format(i + 1), mean)
            tf.summary.histogram('histogram_{}'.format(i + 1), grad)
            tf.summary.histogram('hist_weights_{}'.format(i + 1), grad)

def run_training(model, mnist, sub_folder, iterations=2500, batch_size=30):
    with tf.Session() as sess:
        sess.run(model.init_op)
        train_writer = tf.summary.FileWriter(base_path + sub_folder,
                                             sess.graph)
        for i in range(iterations):
            image_batch, label_batch = mnist.train.next_batch(batch_size)
            l, _, acc = sess.run([model.loss, model.optimizer, model.accuracy],
                                 feed_dict={model.input_images: image_batch, model.labels: label_batch})
            if i % 200 == 0:
                summary = sess.run(model.merged, feed_dict={model.input_images: image_batch,
                                                            model.labels: label_batch})
                train_writer.add_summary(summary, i)
                print("Iteration {} of {}, loss: {:.3f}, train accuracy: "
                      "{:.2f}%".format(i, iterations, l, acc * 100))

if __name__ == "__main__":
    scenarios = ["sigmoid", "relu", "leaky_relu"]
    act_funcs = [tf.sigmoid, tf.nn.relu, tf.nn.leaky_relu]
    assert len(scenarios) == len(act_funcs)
    # collect the training data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    for i in range(len(scenarios)):
        tf.reset_default_graph()
        print("Running scenario: {}".format(scenarios[i]))
        model = Model(784, 10, act_funcs[i], 6, 10)
        run_training(model, mnist, scenarios[i])