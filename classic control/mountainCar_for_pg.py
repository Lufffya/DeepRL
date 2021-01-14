import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(1)


class Agent():
    def __init__(self, n_actions, n_features, max_episode_steps):
        self.n_actions = n_actions
        self.n_features = n_features
        self.max_episode_steps = max_episode_steps
        self.gamma = 0.99
        self.episode_buffer = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.model = self.build_continuous_policy_model()
        self.model.load_weights("model_weights\\mountainCar_for_pg\\pg_checkpoint")
        self.log_std = tf.Variable(tf.zeros(n_actions, dtype=tf.float32))

    def build_continuous_policy_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.n_features,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.n_actions)])
        return model

    def choose_action(self, observation):
        mu = self.model(np.expand_dims(observation, axis=0))
        normal_distribution = tfp.distributions.Normal(mu, tf.math.exp(self.log_std))
        sampled_action = normal_distribution.sample()
        return np.squeeze(sampled_action, axis=0)

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

    def discount_and_norm_rewards(self, rewards):
        # 计算折减奖励
        discounted = np.zeros((len(rewards)))
        discount_rewards = np.power(self.gamma, np.arange(0, len(rewards), dtype=np.float32))
        for t in range(len(rewards)):
            discounted[t] = np.sum(rewards[t:] * discount_rewards[0: len(rewards)-t])
        return discounted

    def train(self):
        states, actions, rewards = self.get_store()
        discounted_rewards = self.discount_and_norm_rewards(rewards)
        with tf.GradientTape() as tape:
            mu = self.model(states)
            normal_distribution = tfp.distributions.Normal(mu, tf.math.exp(self.log_std))
            log_probs = normal_distribution.log_prob(actions)
            loss = - tf.reduce_mean(log_probs * discounted_rewards)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

env = gym.make('MountainCarContinuous-v0')
env.seed(1)
agent = Agent(n_actions=env.action_space.shape[0],
              n_features=env.observation_space.shape[0],
              max_episode_steps=env.spec.max_episode_steps)

i_episode = 0

while True:

    observation = env.reset()

    time_step = 0

    while True:
        env.render()

        action = agent.choose_action(observation)

        observation_, reward, done, _ = env.step(action)

        agent.store(observation, action, reward)

        if done:
            loss = agent.train()
            print(loss.numpy())
            break

        time_step += 1
        observation = observation_

    if i_episode != 0 and i_episode % 10 == 0:
        agent.model.save_weights("model_weights\\mountainCar_for_pg\\pg_checkpoint")

    i_episode += 1
