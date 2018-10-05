import tensorflow as tf
from tensorflow import keras
import datetime as dt

tf.enable_eager_execution()

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# prepare training data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(10000)
train_dataset = train_dataset.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

# prepare validation data
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000)
valid_dataset = valid_dataset.map(lambda x, y: (tf.div(tf.cast(x, tf.float32),255.0), tf.reshape(tf.one_hot(y, 10), (-1, 10))))
valid_dataset = valid_dataset.repeat()

class CIFAR10Model(keras.Model):
    def __init__(self):
        super(CIFAR10Model, self).__init__(name='cifar_cnn')
        self.conv1 = keras.layers.Conv2D(64, 5,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling,
                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.max_pool2d = keras.layers.MaxPooling2D((3, 3), (2, 2), padding='same')
        self.max_norm = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(64, 5,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.variance_scaling,
                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(750, activation=tf.nn.relu,
                                      kernel_initializer=tf.initializers.variance_scaling,
                                      kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.dropout = keras.layers.Dropout(0.5)
        self.fc2 = keras.layers.Dense(10)
        self.softmax = keras.layers.Softmax()

    def call(self, x):
        x = self.max_pool2d(self.conv1(x))
        x = self.max_norm(x)
        x = self.max_pool2d(self.conv2(x))
        x = self.max_norm(x)
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

model = CIFAR10Model()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True)
]

model.fit(train_dataset,  epochs=200, steps_per_epoch=1500,
          validation_data=valid_dataset,
          validation_steps=3, callbacks=callbacks)

