import argparse
import logging
import os
import pickle
import sys
import time
from collections import deque

import cv2
import gym
import numpy as np
import slimevolleygym
import tensorflow as tf
from gym import spaces
from tensorflow.keras import layers

# # Config GPU on home computer
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(funcName)s:%(lineno)d — %(message)s")

"""SLIME ATARI IMPORTS"""


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        (from stable baselines)
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        (from stable-baselines)
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


"""END SLIME ATARI IMPORTS"""


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


def modify_env_slime(env, stack=4):
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = slimevolleygym.FrameStack(env, stack)
    return env


# def modify_env(env, p=1, k=1):
#     def preprocess(S):
#         im = Image.fromarray(np.uint8(S))
#         im = im.convert('YCbCr')
#         Y, Cb, Cr = im.split()
#         Y = Y.resize(size=(84, 84))
#         Y = np.array(Y)
#         Y = Y[..., np.newaxis]
#         return Y
#
#     def new_reset():
#         S = env.orig_reset()
#         return preprocess(S)
#
#     def new_step(A):
#         for i in range(k):
#             S, R, done, info = env.orig_step(A)
#         if np.random.rand() <= p:
#             return preprocess(S), R, done, info
#         else:
#             return np.zeros(shape=(84, 84, 1)), R, done, info
#
#     env.orig_reset = env.reset
#     env.reset = new_reset
#     env.orig_step = env.step
#     env.step = new_step
#     env.p = p
#     return env


class Q_Learn:
    def __init__(self, env, network, max_time_steps, jid, weights=None, clone_steps=10000, batch_size=32, gamma=0.95,
                 epsilon=1.0, epsilonStep=18e-5, eval_X=250, buff_len=1000, initT=0, model=None, buffer=None,
                 render=True):
        self.jid = jid
        if buffer is None:
            self.buffer = deque(maxlen=buff_len)
        else:
            self.buffer = buffer
        self.env = env
        self.network = network
        self.max_time_steps = max_time_steps
        self.clone_steps = clone_steps
        self.batch_size = batch_size
        self.nA = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonStep = epsilonStep
        self.loss_function = tf.keras.losses.mean_squared_error
        # self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.optimizer = tf.keras.optimizers.RMSprop(clipvalue=10.0)
        self.S = None
        self.eval_X = eval_X
        self.X = []
        self.Y = []
        self.t = initT
        self.render = render
        if model is None:
            self.init_model()
        else:
            self.model = model
        if weights is not None:
            logger.info(f"Attempting to load weights from {weights}")
            self.model.load_weights(weights)
            logger.info(f"Successfully loaded weights from {weights}")
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def run(self, eval=True):
        self.S = self.env.reset()
        for _ in range(self.max_time_steps):
            self.t += 1
            if eval and self.t != 0 and self.t % self.eval_X == 0 and self.t > len(self.buffer):
                self.X.append(self.t)
                self.Y.append(self.evaluate())

            if self.render:
                self.env.render()

            if self.t != 0 and self.t % self.clone_steps == 0:
                self.target_model.set_weights(self.model.get_weights())

            self.epsilon = max(0.1, self.epsilon - self.epsilonStep)

            action = self.get_action(self.epsilon)
            logger.info(
                "Step: {}, Action: {}, bufflen: {}, epsilon: {}".format(self.t, action, len(self.buffer), self.epsilon))
            S, R, done, info = self.play_step(action)
            if done:
                self.S = self.env.reset()
            if self.t > len(self.buffer):
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

    def evaluate(self, num_episodes=10, max_episode_time=5):
        values = []
        self.save_run()
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

    def save_run(self):
        model_fpath = os.path.join("results", str(self.jid), f"model_{self.network}_{self.t}.h5")
        fname = os.path.join("results", str(self.jid), f"data_{self.network}_{self.t}.pickle")
        self.model.save(model_fpath)
        data = {"model_fpath": model_fpath,
                "network": self.network,
                "epsilon": self.epsilon,
                "t": self.t,
                "buffer": self.buffer,
                # "env": self.env,
                "maxsteps": self.max_time_steps,
                "epsilonStep": self.epsilonStep,
                "jid": self.jid,
                "evalx": self.eval_X}
        pickle.dump(data, open(fname, "wb"))

    def init_model(self):
        if self.network == "DQN":
            self.model = tf.keras.models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 1)),
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
                layers.Flatten(),
                layers.Dense(units=256, activation="relu"),
                layers.Dense(units=self.nA)
            ])
        elif self.network == "DRQN":
            self.model = tf.keras.models.Sequential([
                layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 1)),
                layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation="relu"),
                layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"),
                layers.Reshape((1, 3136)),
                layers.LSTM(units=256),
                layers.Dense(units=self.nA)
            ])
        else:
            raise ValueError


if __name__ == '__main__':
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-pickle", help="the path to the pickle file ", type=str, default=None)
    parser.add_argument("-render", help="render display", type=bool, default=False)
    parser.add_argument("-env", help="the name of the env to run", type=str, default="SlimeVolleyNoFrameskip-v0")
    parser.add_argument("-network", help="the type of network to load: DQN or DRQN", type=str, default='DQN')
    parser.add_argument("-weights", help="the path to weights to load initially", type=str, default=None)
    parser.add_argument("-initEpsilon", help="the initial epsilon to use", type=float, default=1.0)
    parser.add_argument("-epsilonStep", help="the initial epsilon to use", type=float, default=18e-7)
    parser.add_argument("-initT", help="the initial timestep to use", type=int, default=0)
    parser.add_argument("-jid", help="the slurm job id", type=int)
    parser.add_argument("-bufflen", help="the length of the buffer", type=int, default=10000)
    parser.add_argument("-maxsteps", help="the maximum steps to run", type=int, default=3000000)
    parser.add_argument("-evalx", help="the interval in which to evaluate", type=int, default=2500)
    args = parser.parse_args()
    logger = get_logger(__name__)

    if args.pickle is not None:
        data = pickle.load(open(args.pickle, "rb"))
        model_fpath = data["model_fpath"]
        network = data["network"]
        epsilon = data["epsilon"]
        t = data["t"]
        buffer = data["buffer"]
        # env = data["env"]
        if args.network == "DQN":
            stack = 4
        elif args.network == "DRQN":
            stack = 1
        env = modify_env_slime(gym.make(args.env), stack=stack)
        maxsteps = data["maxsteps"]
        epsilonStep = data["epsilonStep"]
        evalx = data["evalx"]
        jid = data["jid"]
        model = tf.keras.models.load_model(model_fpath)

        logger.info(
            f"Continue training jid={args.jid} with network={network}, env={args.env},  initialEpsilon={epsilon}, epsilonStep={epsilonStep}, initT={t}, max_time_steps={maxsteps}, eval_x={evalx}")

        q_learn = Q_Learn(network=network, env=env, model=model, max_time_steps=maxsteps, epsilon=epsilon,
                          epsilonStep=epsilonStep, eval_X=evalx, initT=t, buffer=buffer, render=args.render, jid=jid)
        q_learn.run(eval=True)
    else:
        results_dir_path = f"results/{args.jid}"
        try:
            os.mkdir(results_dir_path)
        except OSError:
            logger.info("Creation of the directory %s failed" % results_dir_path)
        else:
            logger.info("Successfully created the directory %s" % results_dir_path)

        logger.info(
            f"Begin training jid={args.jid} with network={args.network}, env={args.env}, initialEpsilon={args.initEpsilon}, epsilonStep={args.epsilonStep}, initT={args.initT}, max_time_steps={args.maxsteps}, buff_len={args.bufflen}, eval_x={args.evalx}")

        if args.network == "DQN":
            stack = 4
        elif args.network == "DRQN":
            stack = 1
        env = modify_env_slime(gym.make(args.env), stack=stack)
        q_learn = Q_Learn(env, args.network, args.maxsteps, jid=args.jid, weights=args.weights,
                          epsilon=args.initEpsilon,
                          epsilonStep=args.epsilonStep, initT=args.initT, eval_X=args.evalx, buff_len=args.bufflen,
                          render=args.render)
        model = q_learn.run(eval=True)
        env.close()

        # logger.info("save results (model and scores)")
        # model.save(f'results/{args.jid}/model_{args.network}.h5')
        # np.savetxt(f"results/{args.jid}/GraphA_{args.network}_X.csv", q_learn.X)
        # np.savetxt(f"results/{args.jid}/GraphA_{args.network}_Y.csv", q_learn.Y)
