# -*- coding: utf-8 -*-
# Deep Q Network algorithm of cartpole example
# env: http://gym.openai.com/envs/CartPole-v1/

import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque


class DQN:
    def __init__(self):
        self.lr = 0.003
        self.gamma = 0.99
        self.epsilon = 1
        self.batch_size = 32
        self.buffer = deque(maxlen=1000)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model = self.build_model()

    def build_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(2, activation='linear'),
        ])

    def choose_action(self, state):
        self.epsilon *= 0.995
        self.epsilon = max(self.epsilon, 0.01)

        if np.random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(self.model.predict(np.expand_dims(state, axis=0)))
        return action


    def store(self, state, action, reward, done, next_state):
        self.buffer.append([state, action, reward, done, next_state])
        return len(self.buffer) >= self.batch_size


    def train(self):
        sample_batch = random.sample(self.buffer, self.batch_size)
        for state, action, reward, done, next_state in sample_batch:
            with tf.GradientTape() as tape:
                target = reward if done else reward + self.gamma * self.model(np.expand_dims(next_state, axis=0)).numpy().max()
                y_true = self.model(np.expand_dims(state, axis=0)).numpy()
                y_true[0][action] = target
                loss = tf.keras.losses.mse(y_true,  self.model(np.expand_dims(state, axis=0)))
            gradient = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))



i_episode = 0
env = gym.make("CartPole-v0")
agent = DQN()

while True:

    i_episode += 1
    step_time = 0

    state = env.reset()
    
    while True:
        step_time += 1
        env.render()

        action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)

        if agent.store(state, action, reward, done, next_state):
            agent.train()

        state = next_state

        if done:
            break

    print("i_episode：{} \t step_time：{}".format(i_episode, step_time))
