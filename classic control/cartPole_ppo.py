import sys
import os
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import gym
import torch
import numpy as np
from algorithms.ppo_discrete_torch import PPO, Memory


############## Hyperparameters ##############
env_name = "CartPole-v0"
# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 2
n_latent_var = 16           # number of variables in hidden layer
update_timestep = 512       # update policy every n timesteps
lr = 3e-2
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 10               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = 1
#############################################

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, n_latent_var,lr, betas, gamma, K_epochs, eps_clip)
# ppo.policy_old.load_state_dict(torch.load('weights\\acrobot\\acrobot.pth', map_location='cpu'))
print(lr, betas)

# logging variables
i_episode = 0
total_timestep = 0

# training loop
while True:
    i_episode += 1
    timestep = 0
    running_reward = 0

    state = env.reset()

    while True:
        timestep += 1
        total_timestep += 1

        env.render()

        # Running policy_old:
        action = ppo.policy_old.act(state, memory)
        state, reward, done, _ = env.step(action)

        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if total_timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            total_timestep = 0

        running_reward += reward

        if done:
            break

    print('Episode {} \t reward: {}'.format(i_episode, running_reward))

    # saving
    if i_episode % 500 == 0:
        pass
        # torch.save(ppo.policy.state_dict(), 'weights\\acrobot\\acrobot.pth')