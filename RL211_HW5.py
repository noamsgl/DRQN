from collections import deque

import gym
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def modify_env(env, p=1):
    def new_reset(state=None):
        env.orig_reset()
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)

    env.orig_reset = env.reset
    env.reset = new_reset
    return env
# todo: implement flickering env when p<1


def evaluate_Q_Network(env, model):
    # todo: eval
    raise NotImplementedError


def get_action(S, nA, model, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    else:
        Q_values = model.predict(S[np.newaxis])
        return np.argmax(Q_values[0])


def play_step(env, S, model, epsilon, buffer):
    action = get_action(S, model, epsilon)
    S_tag, R, done, info = env.step(action)
    buffer.append([S, action, R, S_tag])
    return S_tag, R, done, info


def get_experiences(buffer, batch_size):
    idxs = np.random.randint(len(buffer), size=batch_size)
    batch = [buffer[i] for i in idxs]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones


def training_step(batch_size, nA, gamma, model, loss_function, optimizer):
    experiences = get_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1-dones) * gamma * max_next_Q_values)
    mask = tf.one_hot(actions, nA)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_function(target_Q_values, Q_values))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model



def Deep_Q_Learning(env, network, max_time_steps):
    buffer = deque(maxlen=2000)
    epsilon = 0.5
    batch_size = 64
    nA = env.action_space.n
    gamma = 0.95
    if network == "DQN":
        model = keras.models.Sequential([
        #     conv1
        #     conv2
        #     conv3
        #     fc
        #     fc
        ])
    elif network == "DRQN":
        model = keras.models.Sequential([
            #     conv1
            #     conv2
            #     conv3
            #     LSTM
            #     fc
        ])
    else:
        raise ValueError
    loss_function = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(lr=1e-3)

    for t in range(max_time_steps):
        S = env.reset()
        epsilon = 0.99 * epsilon
        S, R, done, info = play_step(env, S, model, epsilon, buffer)
        if done:
            S = env.reset()
        if t > 10000:
            model = training_step(batch_size, nA, gamma, model, loss_function, optimizer)

    return 1


if __name__ == '__main__':

    # graph number 1
    for network in ["DQN", "DRQN"]:
        num_time_steps = 20000
        env = modify_env(gym.make('Frostbite-v0'))
        model = Deep_Q_Learning(env=env, network=network, max_time_steps=num_time_steps)
        X = range(num_time_steps)
        Y = evaluate_Q_Network(env, model)
        plt.plot(X, Y, label=network)

    plt.show()

    #     graph number 2
    for network in ["DQN", "DRQN"]:
        scores = {}
        X = np.linspace(1.0, 0.0, 11, True)
        for p in X:
            env = modify_env(gym.make('Frostbite-v0'), p=p)
            model = Deep_Q_Learning(env=env, network=network, max_time_steps=num_time_steps)
            scores[p] = evaluate_Q_Network(env=env, model=model)
        Y = {k: v / scores[0] for k, v in scores}
        plt.plot(X, Y, label=network)
    plt.show()
