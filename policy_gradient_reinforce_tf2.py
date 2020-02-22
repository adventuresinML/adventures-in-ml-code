import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import datetime as dt

STORE_PATH = '/Users/andrewthomas/Adventures in ML/TensorFlowBook/TensorBoard/PolicyGradientCartPole'
GAMMA = 0.95

env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n

network = keras.Sequential([
    keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(num_actions, activation='softmax')
])
network.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam())

def get_action(network, state, num_actions):
    softmax_out = network(state.reshape((1, -1)))
    selected_action = np.random.choice(num_actions, p=softmax_out.numpy()[0])
    return selected_action


def update_network(network, rewards, states, actions, num_actions):
    reward_sum = 0
    discounted_rewards = []
    for reward in rewards[::-1]:  # reverse buffer r
        reward_sum = reward + GAMMA * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards)
    # standardise the rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    states = np.vstack(states)
    loss = network.train_on_batch(states, discounted_rewards)
    return loss


num_episodes = 10000000
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/PGCartPole_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    states = []
    actions = []
    while True:
        action = get_action(network, state, num_actions)
        new_state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)

        if done:
            loss = update_network(network, rewards, states, actions, num_actions)
            tot_reward = sum(rewards)
            print(f"Episode: {episode}, Reward: {tot_reward}, avg loss: {loss:.5f}")
            with train_writer.as_default():
                tf.summary.scalar('reward', tot_reward, step=episode)
                tf.summary.scalar('avg loss', loss, step=episode)
            break

        state = new_state