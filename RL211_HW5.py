from collections import deque

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers


def preprocess(S):
    im = Image.fromarray(np.uint8(S))
    im = im.convert('YCbCr')
    Y, Cb, Cr = im.split()
    Y = Y.resize(size=(84, 84))
    Y = np.array(Y)
    Y = Y[..., np.newaxis]
    return Y


def modify_env(env, p=1):
    def new_reset():
        S = env.orig_reset()
        return preprocess(S)

    def new_step(A):
        if np.random.rand() <= p:
            S, R, done, info = env.orig_step(A)
            return preprocess(S), R, done, info
        else:
            S, R, done, info = env.orig_step(A)
            return np.zeros(shape=(84, 84, 1)), R, done, info

    env.orig_reset = env.reset
    env.reset = new_reset
    env.orig_step = env.step
    env.step = new_step
    return env


def evaluate_Q_Network(q_learn, num_episodes=100):
    values = []
    for iter in range(num_episodes):
        S = q_learn.env.reset()
        value = 0
        while True:
            A = q_learn.get_action(epsilon=0)
            S, R, done, info = q_learn.env.step(A)
            value += R
            if done:
                break
        values.append(value)
    return np.mean(values)


#
# def get_action(S, nA, model, epsilon=0.0):
#     if np.random.rand() < epsilon:
#         return np.random.randint(nA)
#     else:
#         Q_values = model.predict(S[np.newaxis])
#         return np.argmax(Q_values[0])
#
#
# def play_step(env, S, model, epsilon, buffer):
#     action = get_action(S, env.action_space.n, model, epsilon)
#     S_tag, R, done, info = env.step(action)
#     buffer.append([S, action, R, S_tag])
#     return S_tag, R, done, info
#
#
# def get_experiences(buffer, batch_size):
#     idxs = np.random.randint(len(buffer), size=batch_size)
#     batch = [buffer[i] for i in idxs]
#     states, actions, rewards, next_states, dones = [
#         np.array([experience[field_index] for experience in batch])
#         for field_index in range(5)
#     ]
#     return states, actions, rewards, next_states, dones
#
#
# def training_step(batch_size, nA, gamma, model, loss_function, optimizer, buffer):
#     experiences = get_experiences(buffer, batch_size)
#     states, actions, rewards, next_states, dones = experiences
#     next_Q_values = model.predict(next_states)
#     max_next_Q_values = np.max(next_Q_values, axis=1)
#     target_Q_values = (rewards + (1 - dones) * gamma * max_next_Q_values)
#     mask = tf.one_hot(actions, nA)
#     with tf.GradientTape() as tape:
#         all_Q_values = model(states)
#         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
#         loss = tf.reduce_mean(loss_function(target_Q_values, Q_values))
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return model
#
#
# def Deep_Q_Learning(env, network, max_time_steps):
#     epsilon = 0.5
#     batch_size = 64
#     nA = env.action_space.n
#     gamma = 0.95
#     if network == "DQN":
#         model = tf.keras.models.Sequential([
#             layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 1)),
#             layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
#             layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
#             layers.Flatten(),
#             layers.Dense(units=512, activation="relu"),
#             layers.Dense(units=18)
#         ])
#     elif network == "DRQN":
#         model = tf.keras.models.Sequential([
#             layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 1)),
#             layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
#             layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
#             layers.LSTM(units=512),
#             layers.Dense(units=18)
#         ])
#     else:
#         raise ValueError
#     buffer = deque(maxlen=2000)
#     loss_function = keras.losses.mean_squared_error
#     optimizer = keras.optimizers.Adam(lr=1e-3)
#
#     S = env.reset()
#     for t in range(max_time_steps):
#         env.render()
#         epsilon = 0.99 * epsilon
#         S, R, done, info = play_step(env, S, model, epsilon, buffer)
#         if done:
#             S = env.reset()
#         if t > 10000:
#             model = training_step(batch_size, nA, gamma, model, loss_function, optimizer, buffer)
#
#     return model

class Q_Learn:
    def __init__(self, env, network, max_time_steps, batch_size=64, gamma=0.95, epsilon=0.5):
        self.buffer = deque(maxlen=2000)
        self.env = env
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size
        self.nA = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_function = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.S = None
        if network == "DQN":
            self.model = tf.keras.models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 1)),
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
                layers.Flatten(),
                layers.Dense(units=512, activation="relu"),
                layers.Dense(units=18)
            ])
        elif network == "DRQN":
            self.model = tf.keras.models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 1)),
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
                layers.LSTM(units=512),
                layers.Dense(units=18)
            ])
        else:
            raise ValueError

    def run(self, epsilon=0.5):
        self.S = self.env.reset()
        for t in range(self.max_time_steps):
            self.env.render()
            epsilon = 0.99 * epsilon
            action = self.get_action(epsilon)
            print("Step: {}, Action: {}, bufflen: {}".format(t, action, len(self.buffer)))
            S, R, done, info = self.play_step(action)
            if done:
                self.S = self.env.reset()
            if t > len(self.buffer):
                self.training_step()

        return model

    def play_step(self, action):
        S_tag, R, done, info = self.env.step(action)
        self.buffer.append([self.S, action, R, S_tag, done])
        return S_tag, R, done, info

    def get_action(self, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.nA)
        else:
            Q_values = self.model.predict(self.S[np.newaxis])
            return np.argmax(Q_values[0])

    def training_step(self):
        experiences = self.get_experiences()
        states, actions, rewards, next_states, dones = experiences
        states = states.astype(np.float32)
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * self.gamma * max_next_Q_values)
        mask = tf.one_hot(actions, self.nA)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_function(target_Q_values, Q_values))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_experiences(self):
        idxs = np.random.randint(len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones


if __name__ == '__main__':
    num_time_steps = 200000

    # graph number 1
    for network in ["DQN", "DRQN"]:
        env = modify_env(gym.make('Frostbite-v0'), p=1)
        q_learn = Q_Learn(env, network, num_time_steps)
        model = q_learn.run()
        X = range(num_time_steps)
        Y = evaluate_Q_Network(q_learn)
        plt.plot(X, Y, label=network)
        env.close()

    plt.show()

    #     graph number 2
    for network in ["DQN", "DRQN"]:
        scores = {}
        X = np.linspace(1.0, 0.0, 11, True)
        for p in X:
            env = modify_env(gym.make('Frostbite-v0'), p=p)
            q_learn = Q_Learn(env, network, num_time_steps)
            model = q_learn.run()
            scores[p] = evaluate_Q_Network(q_learn)
            env.close()
        Y = {k: v / scores[0] for k, v in scores}
        plt.plot(X, Y, label=network)
    plt.show()
