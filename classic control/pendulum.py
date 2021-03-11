import sys
import os
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import gym
import torch
import numpy as np
from algorithms.ppo_continuous_torch import PPO, Memory


############## Hyperparameters ##############
env_name = "Pendulum-v0"
render = True
log_interval = 20           # print avg reward in the interval

update_timestep = 512       # update policy every n timesteps
action_std = 0.5            # constant std for action distribution (Multivariate Normal)
K_epochs = 10               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 3e-3                   # parameters for Adam optimizer
betas = (0.9, 0.999)

random_seed = 1
#############################################

# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
ppo.policy_old.load_state_dict(torch.load('weights\\pendulum\\pendulum.pth'))
print(lr, betas)

# logging variables

i_episode = 0
total_timestep = 0
running_reward = 0

# training loop
while True:
    time_step = 0
    i_episode += 1
    state = env.reset()

    while True:
        time_step += 1
        total_timestep += 1

        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if total_timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            total_timestep = 0

        running_reward += reward

        if render:
            env.render()
        if done:
            break

    # save every 500 episodes
    if i_episode % 500 == 0:
        torch.save(ppo.policy.state_dict(), 'weights\\pendulum\\pendulum.pth')

    # logging
    if i_episode % log_interval == 0:
        print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
        running_reward = 0
