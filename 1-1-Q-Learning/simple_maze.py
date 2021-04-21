import numpy as np
import pandas as pd
import time

MAX_EPISODES = 13   # 游戏回合
FRESH_TIME = 0.3    # 每步移动的时长
ALPHA = 0.1         # 学习速率
GAMMA = 0.99        # 折减系数


class Maze_Env():
    def __init__(self):
        self.N_STATES = 6   # 一维迷宫的长度
        self.ACTIONS = ['left', 'right']    # agient 拥有的操作

    def reset(self):
        '''初始化迷宫,并返回初始状态'''
        state = 0
        print('\r{}'.format(''.join(['o'] + (['-']*(self.N_STATES-2)) + ['T'])), end='')
        return state

    def step(self, state, action):
        '''agent 与环境交互'''
        next_state, reward, terminated = None, 0, False

        if action == 'right':   # 向右移动
            if state == self.N_STATES - 2:   # 找到宝藏
                terminated = True
                reward = 1
            else:
                next_state = state + 1
        else:   # 向左移动
            if state == 0:
                next_state = state  # 撞到墙
            else:
                next_state = state - 1

        if not terminated:
            self.update_env(next_state)

        return next_state, reward, terminated

    def update_env(self, next_state):
        '''更新迷宫
        '''
        env_list = ['-']*(self.N_STATES-1) + ['T']
        env_list[next_state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

    def get_n_states(self):
        return self.N_STATES

    def get_actions(self):
        return self.ACTIONS


class Agent():
    def __init__(self, env):
        self.EPSILON = 0.9   # greedy police
        self.n_states = env.get_n_states()
        self.actions = env.get_actions()
        self.q_table = self.build_q_table()

    def build_q_table(self):
        table = pd.DataFrame(
            # q_table 的初始值
            np.zeros((self.n_states, len(self.actions))),
            columns=self.actions,    # q_table 列名为可执行动作名称
        )
        return table

    def choose_action(self, state):
        '''选择动作
        '''
        state_actions = self.q_table.iloc[state, :]  # 得到当前状态的可选择动作
        if (np.random.uniform() > self.EPSILON) or ((state_actions == 0).all()):    # 非随机模式 或者 当前的状态下的Q值无效
            action_name = np.random.choice(self.actions)
        else:   # act greedy
            action_name = state_actions.idxmax()    # 选择对应Q值最大的动作
        return action_name

    def update_q_table(self, state, action, reward, done, next_state):
        q_predict = self.q_table.loc[state, action]

        if done:
            q_target = reward     # 下一个状态是终端
        else:
            q_target = reward + GAMMA * self.q_table.iloc[next_state, :].max()   # 下一个状态不是终端

        self.q_table.loc[state, action] += ALPHA * (q_target - q_predict)  # 更新


if __name__ == "__main__":
    env = Maze_Env()
    agent = Agent(env)

    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = env.reset()  # 初始化环境

        while True:

            action = agent.choose_action(state)

            next_state, reward, done = env.step(state, action)  # 采取行动并获得下一个状态和奖励

            agent.update_q_table(state, action, reward, done, next_state)

            state = next_state  # 移至下一个状态

            step_counter += 1

            if done:
                interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
                print('\r{}'.format(interaction), end='')
                time.sleep(2)
                print('\r                                ', end='')
                break

    print('\r\nQ-table:\n')
    print(agent.q_table)
