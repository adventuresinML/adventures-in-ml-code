import gym
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import datetime as dt
import math

STORE_PATH = '/Users/andrewthomas/Adventures in ML/TensorFlowBook/TensorBoard'
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0005
GAMMA = 0.95
BATCH_SIZE = 32
TAU = 0.08
RANDOM_REWARD_STD = 1.0

env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n

primary_network = keras.Sequential([
    keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(num_actions)
])

target_network = keras.Sequential([
    keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(num_actions)
])

primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

    @property
    def num_samples(self):
        return len(self._samples)


memory = Memory(50000)


def choose_action(state, primary_network, eps):
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(primary_network(state.reshape(1, -1)))


def train(primary_network, memory, target_network=None):
    if memory.num_samples < BATCH_SIZE * 3:
        return 0
    batch = memory.sample(BATCH_SIZE)
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    rewards = np.array([val[2] for val in batch])
    next_states = np.array([(np.zeros(state_size)
                             if val[3] is None else val[3]) for val in batch])
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network(next_states)
    # copy the prim_qt tensor into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt.numpy()
    updates = rewards
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        updates[valid_idxs] += GAMMA * np.amax(prim_qtp1.numpy()[valid_idxs, :], axis=1)
    else:
        prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
        q_from_target = target_network(next_states)
        updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    target_q[batch_idxs, actions] = updates
    loss = primary_network.train_on_batch(states, target_q)
    if target_network is not None:
        # update target network parameters slowly from primary network
        for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
            t.assign(t * (1 - TAU) + e * TAU)
    return loss

num_episodes = 1000
eps = MAX_EPSILON
render = False
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DoubleQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
double_q = False
steps = 0
for i in range(num_episodes):
    state = env.reset()
    cnt = 0
    avg_loss = 0
    while True:
        if render:
            env.render()
        action = choose_action(state, primary_network, eps)
        next_state, reward, done, info = env.step(action)
        reward = np.random.normal(1.0, RANDOM_REWARD_STD)
        if done:
            next_state = None
        # store in memory
        memory.add_sample((state, action, reward, next_state))

        loss = train(primary_network, memory, target_network if double_q else None)
        avg_loss += loss

        state = next_state

        # exponentially decay the eps value
        steps += 1
        eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * steps)

        if done:
            avg_loss /= cnt
            print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.3f}, eps: {eps:.3f}")
            with train_writer.as_default():
                tf.summary.scalar('reward', cnt, step=i)
                tf.summary.scalar('avg loss', avg_loss, step=i)
            break

        cnt += 1


