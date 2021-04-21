import sys, os
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithms.ppo_continuous_tf2 import PPO
from collections import deque
import tensorflow as tf
import numpy as np
import gym
import cv2


########## hyperparameters ##########
GAMMA = 0.99
LAMBDA = 0.95

ACTION_SIZE = 3
EPSILON = 0.2
STEPS_PER_EPOCH = 512
STEPS_PER_BATCH = 128
EPOCHS = 10000000
TRAIN_K_MINIBATCH = 4

LEARNING_RATE = 3e-4
ENTROPY_REG = 0.01
VALUE_COEFFICIENT = 0.5

MAX_GRAD_NORM = 0.5


def compute_gae(rewards, values, bootstrap_values, dones, gamma, lam):
    values = np.vstack((values, [bootstrap_values]))
    deltas = []
    for i in reversed(range(len(rewards))):
        V = rewards[i] + (1.0 - dones[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))

    A = deltas[-1, :]
    advantages = [A]
    for i in reversed(range(len(deltas) - 1)):
        A = deltas[i] + (1.0 - dones[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    advantages = np.array(list(advantages))
    return advantages


def update(ppo, states, actions, returns, advantages, old_log_prob):
    loss = 0
    indexs = np.arange(states.shape[0])
    for i in range(TRAIN_K_MINIBATCH):
        np.random.shuffle(indexs)
        for start in range(0, STEPS_PER_EPOCH, STEPS_PER_BATCH):
            end = start + STEPS_PER_BATCH
            mbinds = indexs[start:end]
            slices = (arr[mbinds] for arr in (states, actions, returns, advantages, old_log_prob))
            pi_loss, value_loss, entropy_loss, total_loss, old_neg_log_val, neg_log_val, approx_kl, ratio = ppo.loss(*slices)

            loss += total_loss.numpy()

    return loss


def train():
    env = gym.make("Pendulum-v0")

    ppo = PPO(ACTION_SIZE, EPSILON, ENTROPY_REG, VALUE_COEFFICIENT,"MLP", LEARNING_RATE, MAX_GRAD_NORM)

    # ppo.load_weights("weights\\CarRacing_PPO\\ppo_checkpoint")

    steps = 0
    total_reward = 0
    timesteps = 0


    state = env.reset()

    for epoch in range(EPOCHS):

        states, actions, values, rewards, dones, old_log_pi = [], [], [], [], [], []

        for t in range(int(STEPS_PER_EPOCH)):

            env.render()

            pi, old_log_p, v = ppo.call(np.expand_dims(state, axis=0))
            pi = pi.numpy()[0]

            clipped_actions = np.clip(pi, env.action_space.low, env.action_space.high)

            next_state, reward, done, _ = env.step(clipped_actions)

            total_reward += reward

            states.append(state)
            actions.append(pi)
            values.append(v.numpy()[0])
            rewards.append(reward)
            dones.append(done)
            old_log_pi.append(old_log_p.numpy()[0])

            if done :
                state = env.reset()

        pi, old_log_p, v = ppo.call(np.expand_dims(next_state, axis=0))
        last_val = v.numpy()[0]

        advantages = compute_gae(rewards, values, last_val, dones, GAMMA, LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        loss = update(ppo, np.array(states), np.array(actions), np.array(returns), np.array(advantages), np.array(old_log_pi))

        print("total_loss：{}   total_reward：{}".format(loss, total_reward))
        # if epoch != 0 and epoch % 10 == 0:
        #     ppo.save_weights("weights\\CarRacing_PPO\\ppo_checkpoint")

    env.close()

if __name__ == "__main__":
    train()
