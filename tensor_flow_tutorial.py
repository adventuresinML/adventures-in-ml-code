import tensorflow as tf
import numpy as np
import datetime as dt
from tensorflow.keras.datasets import mnist

STORE_PATH = '/Users/andrewthomas/Adventures in ML/TensorBoard'

def run_simple_graph():
    # create TensorFlow variables
    const = tf.Variable(2.0, name="const")
    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, name='c')

    # now create some operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d, e, name='a')

    # alternatively (and more naturally)
    d = b + c
    e = c + 2
    a = d * e

    print(f"Variable a is {a.numpy()}")


def run_simple_graph_multiple():
    const = tf.Variable(2.0, name="const")
    b = tf.Variable(np.arange(0, 10), name='b')
    c = tf.Variable(1.0, name='c')

    d = tf.cast(b, tf.float32) + c
    e = c + const
    a = d * e

    print(f"Variable a is {a.numpy()}")

    # the line below would cause an error - tensors are immutable
    # b[1] = 10

    # need to use assignment instead
    b[1].assign(10)
    d = tf.cast(b, tf.float32) + c
    e = c + const
    a = d * e
    print(f"Variable a is {a.numpy()}")

    b[6:9].assign([10, 10, 10])
    f = b[2:5]
    print(f.numpy())


def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]


def nn_model(x_input, W1, b1, W2, b2):
    # flatten the input image from 28 x 28 to 784
    x_input = tf.reshape(x_input, (x_input.shape[0], -1))
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, W2), b2)
    return logits


def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                              logits=logits))
    return cross_entropy


def nn_example():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Python optimisation variables
    epochs = 10
    batch_size = 100

    # normalize the input images by dividing by 255.0
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # convert x_test to tensor to pass through model (train data will be converted to
    # tensors on the fly)
    x_test = tf.Variable(x_test)

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random.normal([10]), name='b2')

    # setup the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # create a summary writer to view loss in TensorBoard
    train_summary_writer = tf.summary.create_file_writer(STORE_PATH +
                                                         "/TensorFlow_Intro_Chapter_" +
                                                         f"{dt.datetime.now().strftime('%d%m%Y%H%M')}")

    total_batch = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        avg_loss = 0
        for i in range(total_batch):
            batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
            # create tensors
            batch_x = tf.Variable(batch_x)
            batch_y = tf.Variable(batch_y)
            # create a one hot vector
            batch_y = tf.one_hot(batch_y, 10)
            with tf.GradientTape() as tape:
                logits = nn_model(batch_x, W1, b1, W2, b2)
                loss = loss_fn(logits, batch_y)
            gradients = tape.gradient(loss, [W1, b1, W2, b2])
            optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
            avg_loss += loss / total_batch
        test_logits = nn_model(x_test, W1, b1, W2, b2)
        max_idxs = tf.argmax(test_logits, axis=1)
        test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
        print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set accuracy={test_acc*100:.3f}%")
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('accuracy', test_acc, step=epoch)


    print("\nTraining complete!")

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    nn_example()