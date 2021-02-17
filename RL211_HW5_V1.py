import argparse
import logging
import os
import sys
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_console_handler())
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


def modify_env(env, p=1, k=4):
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
        for i in range(k):
            S, R, done, info = env.orig_step(A)
        if np.random.rand() <= p:
            return preprocess(S), R, done, info
        else:
            return np.zeros(shape=(84, 84, 1)), R, done, info

    env.p = p
    env.orig_reset = env.reset
    env.reset = new_reset
    env.orig_step = env.step
    env.step = new_step
    return env


class Q_Learn:
    def __init__(self, env, network, max_time_steps, clone_steps=10000, batch_size=32, gamma=0.95, epsilon=1.0,
                 eval_X=250, buff_len=1000, render=True):
        self.buffer = deque(maxlen=buff_len)
        self.env = env
        self.network = network
        self.max_time_steps = max_time_steps
        self.clone_steps = clone_steps
        self.batch_size = batch_size
        self.nA = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_function = tf.keras.losses.mean_squared_error
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue=10.0)
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
                layers.Dense(units=self.nA)
            ])
        elif network == "DRQN":
            self.model = tf.keras.models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 1)),
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
                layers.Reshape((1, 3136)),
                layers.LSTM(units=512),
                layers.Dense(units=self.nA)
            ])
        else:
            raise ValueError
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def run(self, eval=False):
        self.S = self.env.reset()
        for t in range(self.max_time_steps):
            if eval and t != 0 and t % self.eval_X == 0 and t > len(self.buffer):
                self.X.append(t)
                self.Y.append(self.evaluate(t=t))

            if self.render:
                self.env.render()

            if t != 0 and t % self.clone_steps == 0:
                self.target_model.set_weights(self.model.get_weights())

            self.epsilon = max(0.1, self.epsilon - 18e-7)

            action = self.get_action(self.epsilon)
            logger.info(
                "Step: {}, Action: {}, bufflen: {}, epsilon: {}".format(t, action, len(self.buffer), self.epsilon))
            S, R, done, info = self.play_step(action)
            if done:
                self.S = self.env.reset()
            if t > len(self.buffer):
                self.training_step()
        return self.model

    def play_step(self, action):
        S_tag, R, done, info = self.env.step(action)
        R = self.clip_reward(R)
        self.buffer.append([self.S, action, R, S_tag, done])
        return S_tag, R, done, info

    def clip_reward(self, R):
        if R > 1:
            return 1
        elif R < -1:
            return -1
        else:
            return R

    def get_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.nA)
        else:
            Q_values = self.model.predict(self.S[np.newaxis])
            return np.argmax(Q_values[0])

    def training_step(self):
        experiences = self.get_experiences()
        states, actions, rewards, next_states, dones = experiences
        states = states.astype(np.float32)
        next_Q_values = self.target_model.predict(next_states)
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

    def evaluate(self, num_episodes=10, max_episode_time=5, t=None):
        values = []
        if t is not None:
            self.model.save_weights(os.path.join(results_dir_path, f"weights_{self.network}_{self.env.p}_{t}.h5"))
        for iter in range(num_episodes):
            self.env.reset()
            value = 0
            start_time = time.time()
            done = False
            while (time.time() - start_time) < max_episode_time * 60 and not done:
                if self.render:
                    self.env.render()
                A = self.get_action(epsilon=0.0)
                S, R, done, info = self.env.step(A)
                value += R

            logger.info("Evaluating: iter {} of {}, Value: {}".format(iter, num_episodes, value))
            values.append(value)
        return np.mean(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="the 1-probability for the env to flicker", type=float, default=1)
    parser.add_argument("-env", help="the name of the env to run", type=str, default='Frostbite-v0')
    parser.add_argument("-jid", help="the slurm job id", type=int)
    args = parser.parse_args()
    args.p = 1.0 - args.p
    logger = get_logger(__name__)
    render = False
    np.random.seed(0)
    max_time_steps = 10000000
    buff_len = 40000
    eval_X = 2500
    results_dir_path = f"results/{args.jid}"
    try:
        os.mkdir(results_dir_path)
    except OSError:
        logger.info("Creation of the directory %s failed" % results_dir_path)
    else:
        logger.info("Successfully created the directory %s" % results_dir_path)

    for network in ["DQN", "DRQN"]:
        logger.info(
            f"Begin training jid={args.jid} with network={network}, p={args.p}, env={args.env}, max_time_steps={max_time_steps}, buff_len={buff_len}, eval_x={eval_X}")
        env = modify_env(gym.make(args.env), p=args.p)
        q_learn = Q_Learn(env, network, max_time_steps, eval_X=eval_X, buff_len=buff_len, render=render)
        model = q_learn.run(eval=True)
        env.close()

        logger.info("save results (model and scores)")
        model.save(f'results/{args.jid}/model_{network}_{args.p}.h5')
        np.savetxt(f"results/{args.jid}/GraphA_{network}_{args.p}_X.csv", q_learn.X)
        np.savetxt(f"results/{args.jid}/GraphA_{network}_{args.p}_Y.csv", q_learn.Y)
