import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

STORE_PATH = 'C:\\Users\\Andy\\TensorFlowBook\\TensorBoard'

def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]

def nn_example():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Python optimisation variables
    learning_rate = 0.5
    epochs = 20
    batch_size = 100

    with tf.name_scope("inputs"):
        # declare the training data placeholders
        x = tf.placeholder(tf.float32, [None, 28, 28])
        # reshape input x - for 28 x 28 pixels = 784
        x_rs = tf.reshape(x, [-1, 784])
        # scale the input data (maximum is 255.0, minimum is 0.0)
        x_sc = tf.div(x_rs, 255.0)
        # now declare the output data placeholder - 10 digits
        y = tf.placeholder(tf.int64, [None, 1])
        # convert the y data to one hot values
        y_one_hot = tf.reshape(tf.one_hot(y, 10), [-1, 10])

    with tf.name_scope("layer_1"):
        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.01), name='W')
        b1 = tf.Variable(tf.random_normal([300]), name='b')
        hidden_logits = tf.add(tf.matmul(x_sc, W1), b1)
        hidden_out = tf.nn.sigmoid(hidden_logits)
        tf.summary.histogram("Hidden_logits", hidden_logits)
        tf.summary.histogram("Hidden_output", hidden_out)
    with tf.name_scope("layer_2"):
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.05), name='W')
        b2 = tf.Variable(tf.random_normal([10]), name='b')
        logits = tf.add(tf.matmul(hidden_out, W2), b2)


    # now let's define the cost function which we are going to train the model on
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot,
                                                            logits=logits))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_one_hot, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope("getimages"):
        correct_inputs = tf.boolean_mask(x_sc, correct_prediction)
        image_summary_true = tf.summary.image('correct_images', tf.reshape(correct_inputs, (-1, 28, 28, 1)),
                                              max_outputs=5)
        incorrect_inputs = tf.boolean_mask(x_sc, tf.logical_not(correct_prediction))
        image_summary_false = tf.summary.image('incorrect_images', tf.reshape(incorrect_inputs, (-1, 28, 28, 1)),
                                               max_outputs=5)

    # add a summary to store the accuracy
    tf.summary.scalar('acc_summary', accuracy)

    merged = tf.summary.merge_all()
    # start the session
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter(STORE_PATH, sess.graph)
        # initialise the variables
        total_batch = int(len(y_train) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y.reshape(-1, 1)})
                avg_cost += c / total_batch
            acc, summary = sess.run([accuracy, merged], feed_dict={x: x_test, y: y_test.reshape(-1, 1)})
            print("Epoch: {}, cost={:.3f}, test set accuracy={:.3f}%".format(epoch + 1, avg_cost, acc*100))
            writer.add_summary(summary, epoch)
        print("\nTraining complete!")

if __name__ == "__main__":
    nn_example()