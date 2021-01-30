import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import datetime as dt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


STORE_PATH = '/Users/andrewthomas/Adventures in ML/TensorFlowBook/TensorBoard/A2CCartPole'
CRITIC_LOSS_WEIGHT = 0.5
ACTOR_LOSS_WEIGHT = 1.0
ENTROPY_LOSS_WEIGHT = 0.05
BATCH_SIZE = 64
GAMMA = 0.95

env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n


class Model(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.value = keras.layers.Dense(1)
        self.policy_logits = keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, state):
        value, logits = self.predict_on_batch(state)
        action = tf.random.categorical(logits, 1)[0]
        return action, value


def critic_loss(discounted_rewards, predicted_values):
    return keras.losses.mean_squared_error(discounted_rewards, predicted_values) * CRITIC_LOSS_WEIGHT


def actor_loss(combined, policy_logits):
    actions = combined[:, 0]
    advantages = combined[:, 1]
    sparse_ce = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    )

    actions = tf.cast(actions, tf.int32)
    policy_loss = sparse_ce(actions, policy_logits, sample_weight=advantages)

    probs = tf.nn.softmax(policy_logits)
    entropy_loss = keras.losses.categorical_crossentropy(probs, probs)

    return policy_loss * ACTOR_LOSS_WEIGHT - entropy_loss * ENTROPY_LOSS_WEIGHT


def discounted_rewards_advantages(rewards, dones, values, next_value):
    discounted_rewards = np.array(rewards + [next_value[0]])

    for t in reversed(range(len(rewards))):
        discounted_rewards[t] = rewards[t] + GAMMA * discounted_rewards[t+1] * (1-dones[t])
    discounted_rewards = discounted_rewards[:-1]
    # advantages are bootstrapped discounted rewards - values, using Bellman's equation
    advantages = discounted_rewards - np.stack(values)[:, 0]
    return discounted_rewards, advantages


model = Model(num_actions)
model.compile(optimizer=keras.optimizers.Adam(), loss=[critic_loss, actor_loss])

train_writer = tf.summary.create_file_writer(STORE_PATH + f"/A2C-CartPole_{dt.datetime.now().strftime('%d%m%Y%H%M')}")

num_steps = 10000000
episode_reward_sum = 0
state = env.reset()
episode = 1
for step in range(num_steps):
    rewards = []
    actions = []
    values = []
    states = []
    dones = []
    for _ in range(BATCH_SIZE):
        _, policy_logits = model(state.reshape(1, -1))

        action, value  = model.action_value(state.reshape(1, -1))
        new_state, reward, done, _ = env.step(action.numpy()[0])

        actions.append(action)
        values.append(value.numpy()[0])
        states.append(state)
        dones.append(done)
        episode_reward_sum += reward

        state = new_state
        if done:
            rewards.append(0.0)
            state = env.reset()
            print(f"Episode: {episode}, latest episode reward: {episode_reward_sum}, loss: {loss}")
            with train_writer.as_default():
                tf.summary.scalar('rewards', episode_reward_sum, episode)
            episode_reward_sum = 0
            episode += 1
        else:
            rewards.append(reward)

    _, next_value = model.action_value(state.reshape(1, -1))
    discounted_rewards, advantages = discounted_rewards_advantages(rewards, dones, values, next_value.numpy()[0])

    # combine the actions and advantages into a combined array for passing to
    # actor_loss function
    combined = np.zeros((len(actions), 2))
    combined[:, 0] = actions
    combined[:, 1] = advantages

    loss = model.train_on_batch(tf.stack(states), [discounted_rewards, combined])

    with train_writer.as_default():
        tf.summary.scalar('tot_loss', np.sum(loss), step)