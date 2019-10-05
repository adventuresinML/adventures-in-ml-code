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
EPSILON_MIN_ITER = 5000
DELAY_TRAINING = 300
GAMMA = 0.95
BATCH_SIZE = 32
TAU = 0.08
RANDOM_REWARD_STD = 1.0

env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n


class DQModel(keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, dueling: bool):
        super(DQModel, self).__init__()
        self.dueling = dueling
        self.dense1 = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_dense = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(num_actions,
                                          kernel_initializer=keras.initializers.he_normal())
        if dueling:
            self.v_dense = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
            self.v_out = keras.layers.Dense(1, kernel_initializer=keras.initializers.he_normal())
            self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            self.combine = keras.layers.Add()

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        if self.dueling:
            v = self.v_dense(x)
            v = self.v_out(v)
            norm_adv = self.lambda_layer(adv)
            combined = self.combine([v, norm_adv])
            return combined
        return adv

primary_network = DQModel(30, num_actions, True)
target_network = DQModel(30, num_actions, True)
primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')
# make target_network = primary_network
for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
    t.assign(e)

def update_network(primary_network, target_network):
    # update target network parameters slowly from primary network
    for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
        t.assign(t * (1 - TAU) + e * TAU)

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


memory = Memory(500000)


def choose_action(state, primary_network, eps):
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(primary_network(state.reshape(1, -1)))


def train(primary_network, memory, target_network):
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
    # extract the best action from the next state
    prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
    # get all the q values for the next state
    q_from_target = target_network(next_states)
    # add the discounted estimated reward from the selected action (prim_action_tp1)
    updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    # update the q target to train towards
    target_q[batch_idxs, actions] = updates
    # run a training batch
    loss = primary_network.train_on_batch(states, target_q)
    return loss


num_episodes = 1000000
eps = MAX_EPSILON
render = False
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DuelingQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
steps = 0
for i in range(num_episodes):
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    state = env.reset()
    while True:
        if render:
            env.render()
        action = choose_action(state, primary_network, eps)
        next_state, _, done, info = env.step(action)
        reward = np.random.normal(1.0, RANDOM_REWARD_STD)
        tot_reward += reward
        if done:
            next_state = None
        # store in memory
        memory.add_sample((state, action, reward, next_state))

        if steps > DELAY_TRAINING:
            loss = train(primary_network, memory, target_network)
            update_network(primary_network, target_network)
        else:
            loss = -1
        avg_loss += loss

        # linearly decay the eps value
        if steps > DELAY_TRAINING:
            eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * \
                  (MAX_EPSILON - MIN_EPSILON) if steps < EPSILON_MIN_ITER else \
                MIN_EPSILON
        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss /= cnt
                print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', cnt, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
            else:
                print(f"Pre-training...Episode: {i}")
            break

        state = next_state
        cnt += 1

