import argparse
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers


def modify_env(env, p=1):
    def preprocess(S):
        im = Image.fromarray(np.uint8(S))
        im = im.convert('YCbCr')
        Y, Cb, Cr = im.split()
        Y = Y.resize(size=(84, 84))
        Y = np.array(Y)
        Y = Y[..., np.newaxis]
        return Y

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


class Q_Learn:
    def __init__(self, env, network, max_time_steps, batch_size=64, gamma=0.95, epsilon=0.5, eval_X=250, buff_len=1000,
                 render=True):
        self.buffer = deque(maxlen=buff_len)
        self.env = env
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size
        self.nA = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_function = tf.keras.losses.mean_squared_error
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.S = None
        self.eval_X = eval_X
        self.X = []
        self.Y = []
        self.render = render
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
                layers.Reshape((1, 3136)),
                layers.LSTM(units=512),
                layers.Dense(units=18)
            ])
        else:
            raise ValueError

    def run(self, epsilon=0.5, eval=False):
        self.S = self.env.reset()
        for t in range(self.max_time_steps):
            if eval and t != 0 and t % self.eval_X == 0:
                self.X.append(t)
                self.Y.append(self.evaluate())
            if self.render:
                self.env.render()

            epsilon = 0.999 * epsilon
            action = self.get_action(epsilon)
            print("Step: {}, Action: {}, bufflen: {}, epsilon: {}".format(t, action, len(self.buffer), epsilon))
            S, R, done, info = self.play_step(action)
            if done:
                self.S = self.env.reset()
            if t > len(self.buffer):
                self.training_step()
        return self.model

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

    def evaluate(self, num_episodes=10, max_episode_len=2000):
        values = []
        for iter in range(num_episodes):
            self.env.reset()
            value = 0
            for i in range(max_episode_len):
                if self.render:
                    self.env.render()
                A = self.get_action(epsilon=0)
                S, R, done, info = self.env.step(A)
                value += R
                if done:
                    break
            print("Evaluating: iter {} of {}, Value: {}".format(iter, num_episodes, value))
            values.append(value)
        return np.mean(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="the probability for the env to flicker", type=float, default=1)
    parser.add_argument("-env", help="the name of the env to run", type=str, default='Frostbite-v0')
    args = parser.parse_args()
    render = False
    np.random.seed(0)
    max_time_steps = 20000
    buff_len = 1000

    for network in ["DRQN", "DQN"]:
        print("Begin training with network {}, p={}".format(network, args.p))
        env = modify_env(gym.make(args.env), p=args.p)
        q_learn = Q_Learn(env, network, max_time_steps, eval_X=500, buff_len=buff_len, render=render)
        model = q_learn.run(eval=True)
        env.close()

        # save results (model and scores)
        model.save('results/model_{}_{}.h5'.format(network, args.p))
        np.savetxt("results/GraphA_{}_{}_X.csv".format(network, args.p), q_learn.X)
        np.savetxt("results/GraphA_{}_{}_Y.csv".format(network, args.p), q_learn.Y)
