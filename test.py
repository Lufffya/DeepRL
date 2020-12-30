import numpy as np
import tensorflow as tf

np.random.seed(1)


class Agent:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.99):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.model = self.build_Model()
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.episode_buffer = []

    def build_Model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation="tanh"),
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
            states.append(states)
            actions.append(action)
            rewards.append(reward)
        self.episode_buffer = []

        return np.array(states), np.array(actions), np.array(rewards)

    def train(self):
        states, actions, rewards = self.get_store()
        discounted_rewards = self.discount_and_norm_rewards(rewards)

        with tf.GradientTape() as tape:

            actions_prob = self.model(states)

            neg_log_prob = tf.reduce_sum(-tf.log(actions_prob) * tf.one_hot(actions, self.n_actions), axis=1)

            loss = tf.reduce_mean(neg_log_prob * discounted_rewards)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def discount_and_norm_rewards(self, rewards):
        # discounted episode rewards
        discount_rewards = []
        discounted_sum = 0
        for reward in rewards[::-1]:
            discounted_sum = discounted_sum * self.gamma + reward
            discount_rewards.append(discounted_sum)
        discount_rewards = discount_rewards[::-1]

        # normalize episode rewards
        normalize_rewards = (
            discount_rewards - np.mean(discount_rewards)) / np.std(discount_rewards)

        return normalize_rewards




import gym
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = Agent(n_actions=env.action_space.n,n_features=env.observation_space.shape[0],learning_rate=0.01,reward_decay=0.99,)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = agent.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        agent.store(observation, action, reward)

        if done:
            agent.train()
            break
            
        observation = observation_
