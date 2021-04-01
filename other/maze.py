import gym
import gym_maze
import numpy as np


lr = 0.2
gamma = 0.99
epsilon = 0.9
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

# actions
# ["N", "S", "E", "W"]
# ["北", "南", "东", "西"]
# ["上", "下", "右", "左"]

Q_Table = np.zeros((20, 20, 4))

# Q_Table = np.load(file="maze_model.npy")

i_episode = 0

while True:

    time_step = 0
    i_episode += 1

    state = env.reset()
    state = tuple([int(item) for item in state])

    while True:
        time_step += 1

        if np.any(Q_Table[state]) == 0:
            action = np.random.randint(0, 3)
        else:
            action = int(np.argmax(Q_Table[state]))

        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)

        env.render()

        Q_Table[state][action] += lr * ((reward + gamma * np.amax(Q_Table[next_state])) - Q_Table[state][action])

        state = next_state

        if done:
            print("第 {} 次游戏, 在第 {} 步尝试后到达终点！".format(i_episode, time_step))
            break
    
    # np.save(file="maze_model.npy", arr=Q_Table)