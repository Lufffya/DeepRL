import gym
import gym_maze
import math
import numpy as np
import random

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
# env_name = "maze-random-10x10-plus-v0"
env_name = "maze-random-20x20-plus-v0"
# env_name = "maze-random-30x30-plus-v0"

env = gym.make(env_name)

MIN_LEARNING_RATE = 0.2
MIN_EPSILON_RATE = 0.001
MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_epsilon_rate(t):
    return max(MIN_EPSILON_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(Q_Table[state]))
    return action


# actions
# ["N", "S", "E", "W"]
# ["北", "南", "东", "西"]
# ["上", "下", "右", "左"]

Q_Table = np.zeros((20, 20, 4))

# Q_Table = np.load(file="Weights\maze_model.npy")

i_episode = 0
lr = get_learning_rate(0)
epsilon = get_epsilon_rate(0)
gamma = 0.99

while True:

    time_step = 0
    i_episode += 1

    state = env.reset()
    state = tuple([int(item) for item in state])

    while True:
        time_step += 1

        action = select_action(state, epsilon)

        # if np.any(Q_Table[state]) == 0:
        #     action = np.random.randint(0, 3)
        # else:
        #     action = int(np.argmax(Q_Table[state]))

        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)

        env.render()

        Q_Table[state][action] += lr * ((reward + gamma * np.amax(Q_Table[next_state])) - Q_Table[state][action])

        state = next_state

        if done:
            print("第 {} 次游戏, 在第 {} 步尝试后到达终点！".format(i_episode, time_step))
            break
    
    lr = get_learning_rate(i_episode)
    epsilon = get_epsilon_rate(i_episode)

    # np.save(file="Weights\maze_model.npy", arr=Q_Table)
