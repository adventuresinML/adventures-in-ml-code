import os
os.environ['PATH'] = "C:\\Users\Andy\Anaconda2\envs\TensorFlow" + ';' + os.environ['PATH']

import cntk as C
from cntk.train import Trainer
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk.learners import adadelta, learning_rate_schedule, UnitType
from cntk.ops import relu, element_times, constant
from cntk.layers import Dense, Sequential, For, default_options
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.train.training_session import *
from cntk.logging import ProgressPrinter

abs_path = os.path.dirname(os.path.abspath(__file__))

# Creates and trains a feedforward classification model for MNIST images
def simple_mnist():
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 2
    hidden_layers_dim = 200

    # Input variables denoting the features and label data
    feature = C.input_variable(input_dim)
    label = C.input_variable(num_output_classes)

    # Instantiate the feedforward classification model
    scaled_input = element_times(constant(0.00390625), feature)

    # z = Sequential([
    #     Dense(hidden_layers_dim, activation=relu),
    #     Dense(hidden_layers_dim, activation=relu),
    #     Dense(num_output_classes)])(scaled_input)

    with default_options(activation=relu, init=C.glorot_uniform()):
        z = Sequential([For(range(num_hidden_layers),
            lambda i: Dense(hidden_layers_dim)),
            Dense(num_output_classes, activation=None)])(scaled_input)

    ce = cross_entropy_with_softmax(z, label)
    pe = classification_error(z, label)

    # setup the data
    path = abs_path + "\Train-28x28_cntk_text.txt"

    reader_train = MinibatchSource(CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=input_dim),
        labels=StreamDef(field='labels', shape=num_output_classes))))

    input_map = {
        feature: reader_train.streams.features,
        label: reader_train.streams.labels
    }

    # Training config
    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10

    # Instantiate progress writers.
    progress_writers = [ProgressPrinter(
        tag='Training',
        num_epochs=num_sweeps_to_train_with)]

    # Instantiate the trainer object to drive the model training
    lr = learning_rate_schedule(1, UnitType.sample)
    trainer = Trainer(z, (ce, pe), [adadelta(z.parameters, lr)], progress_writers)

    training_session(
        trainer=trainer,
        mb_source=reader_train,
        mb_size=minibatch_size,
        model_inputs_to_streams=input_map,
        max_samples=num_samples_per_sweep * num_sweeps_to_train_with,
        progress_frequency=num_samples_per_sweep
    ).train()

    # Load test data
    path = abs_path + "\Test-28x28_cntk_text.txt"

    reader_test = MinibatchSource(CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=input_dim),
        labels=StreamDef(field='labels', shape=num_output_classes))))

    input_map = {
        feature: reader_test.streams.features,
        label: reader_test.streams.labels
    }

    # Test data for trained model
    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=input_map)
        eval_error = trainer.test_minibatch(mb)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    return test_result / num_minibatches_to_test


if __name__ == '__main__':
    error = simple_mnist()
    print("Error: %f" % error)

