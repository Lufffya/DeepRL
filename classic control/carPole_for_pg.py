import gym
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(42)


class Agent:
    def __init__(self, n_actions, n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = 0.99
        self.model = self.build_Policy_Model()
        self.optimizer = tf.keras.optimizers.Adam(lr=0.01)
        self.episode_buffer = []

    def build_Policy_Model(self):
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
            loss = discounted_rewards * loss

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def discount_and_norm_rewards(self, rewards):
        # 计算折减奖励
        discount_rewards = []
        discounted_sum = 0
        for reward in rewards:
            discounted_sum = discounted_sum * self.gamma + reward
            discount_rewards.append(discounted_sum)
        discount_rewards = discount_rewards[::-1]

        # # 标准化奖励
        normalize_rewards = discount_rewards - np.mean(discount_rewards)

        return normalize_rewards

env = gym.make('CartPole-v0')
env.seed(42)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = Agent(n_actions=env.action_space.n,
              n_features=env.observation_space.shape[0])

for i_episode in range(3000):

    observation = env.reset()

    time_step = 0

    while True:
        env.render()

        action = agent.choose_action(observation)

        observation_, reward, done, _ = env.step(action)

        agent.store(observation, action, reward)

        if done:
            loss = agent.train()
            print(loss.numpy(), time_step)
            break

        time_step += 1
        observation = observation_
