# -*- coding: utf-8 -*-
# Q-Learning algorithm of maze example
# env: https://github.com/MattChanTK/gym-maze

import gym
import gym_maze
import math
import random
import numpy as np

###### default ######
# env_name = "maze-v0"
###### sample ######
# env_name = "maze-sample-3x3-v0"
# env_name = "maze-sample-5x5-v0"
# env_name = "maze-sample-10x10-v0"
# env_name = "maze-sample-100x100-v0"
###### random ######
# env_name = "maze-random-3x3-v0"
# env_name = "maze-random-5x5-v0"
# env_name = "maze-random-10x10-v0"
# env_name = "maze-random-100x100-v0"
###### plus ######
env_name = "maze-random-10x10-plus-v0"
# env_name = "maze-random-20x20-plus-v0"
# env_name = "maze-random-30x30-plus-v0"
###### actions ######
# ["N", "S", "E", "W"]
# ["北", "南", "东", "西"]
# ["上", "下", "右", "左"]

env = gym.make(env_name)


class Q_Learning:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.MIN_LEARNING_RATE = 0.2
        self.MIN_EPSILON_RATE = 0.001
        self.DECAY_FACTOR = np.prod(tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int)), dtype=float) / 10.0
        self.epsilon_rate = self.get_epsilon_rate(0)
        self.learning_rate = self.get_learning_rate(0)
        self.Q_Table = np.zeros((20, 20, 4))
        # self.Q_Table = np.load(file="Weights\maze_model.npy")

    def get_learning_rate(self, i_episode):
        return max(self.MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((i_episode+1)/self.DECAY_FACTOR)))

    def get_epsilon_rate(self, i_episode):
        return max(self.MIN_EPSILON_RATE, min(0.8, 1.0 - math.log10((i_episode+1)/self.DECAY_FACTOR)))

    def choose_action(self, state):
        # if np.any(Q_Table[state]) == 0:
        #     action = np.random.randint(0, 3)
        # else:
        #     action = int(np.argmax(Q_Table[state]))

        if random.random() < self.epsilon_rate:
            action = self.env.action_space.sample()
        else:
            action = int(np.argmax(self.Q_Table[state]))
        return action

    def train(self, state, action, reward, next_state):
        self.Q_Table[state][action] += self.learning_rate * ((reward + self.gamma * np.amax(self.Q_Table[next_state])) - self.Q_Table[state][action])



i_episode = 0
agent = Q_Learning(env)

while True:

    time_step = 0
    i_episode += 1

    state = env.reset()
    state = tuple([int(item) for item in state])

    while True:
        time_step += 1

        env.render()

        action = agent.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)

        agent.train(state, action, reward, next_state)

        state = next_state

        if done:
            print("第 {} 次游戏, 在第 {} 步尝试后到达终点！".format(i_episode, time_step))
            break
    
    agent.learning_rate = agent.get_learning_rate(i_episode)
    agent.epsilon_rate = agent.get_epsilon_rate(i_episode)

    # np.save(file="Weights\maze_model.npy", arr=Q_Table)
