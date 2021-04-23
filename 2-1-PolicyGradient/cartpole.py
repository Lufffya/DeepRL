# -*- coding: utf-8 -*-
# Discrete Policy Gradient algorithm of cartpole example
# env: http://gym.openai.com/envs/CartPole-v1/

import gym
import numpy as np
import tensorflow as tf


class PG:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 0.99
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(lr=0.01)
        self.episode_buffer = []

    def build_model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, activation="relu", input_shpe=(None, self.n_features)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(self.n_actions, activation="softmax")
        ])

    def choose_action(self, observation):
        actions_prob = self.model(np.expand_dims(observation, 0))
        action = np.random.choice(self.n_actions, p=np.squeeze(actions_prob))
        return action

    def store(self, state, action, reward):
        self.episode_buffer.append((state, action, reward))

    def get_store(self):
        states, actions, rewards = [], [], []
        for state, action, reward in self.episode_buffer:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        self.episode_buffer = []
        return np.array(states), np.array(actions), np.array(rewards)

    def train(self):
        states, actions, rewards = self.get_store()
        discounted_rewards = self.discount_and_norm_rewards(rewards)
        with tf.GradientTape() as tape:
            actions_prob = self.model(states)
            loss = tf.keras.losses.categorical_crossentropy(y_pred=actions_prob, y_true=tf.one_hot(actions, self.n_actions))
            loss = tf.reduce_mean(discounted_rewards * loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def discount_and_norm_rewards(self, rewards):
        # 计算折减奖励
        discount_rewards = []
        discounted_sum = 0
        for reward in rewards:
            discounted_sum = discounted_sum * self.gamma + reward
            discount_rewards.append(discounted_sum)
        discount_rewards = discount_rewards[::-1]
        # 标准化奖励
        normalize_rewards = (discount_rewards - np.mean(discount_rewards)) / np.std(discount_rewards)
        return normalize_rewards


env = gym.make('CartPole-v1')
agent = PG(n_actions=env.action_space.n, n_features=env.observation_space.shape[0])
i_episode = 0

while True:
    time_step = 0
    i_episode += 1

    observation = env.reset()

    while True:
        time_step += 1
        env.render()

        action = agent.choose_action(observation)

        observation_, reward, done, _ = env.step(action)

        agent.store(observation, action, reward)

        if done:
            agent.train()
            print("i_episode: {0} \t time_step: {2}".format(i_episode, time_step))
            break

        observation = observation_
