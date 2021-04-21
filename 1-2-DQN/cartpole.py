import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque


lr = 0.003
gamma = 0.99
epsilon = 1
batch_size = 32
buffer = deque(maxlen=1000)
env = gym.make("CartPole-v0")
optimizer = tf.keras.optimizers.Adam(lr)


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(None, 4)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2, activation='linear'),
    ])


def select_action(state):
    global epsilon
    epsilon *= 0.995
    epsilon = max(epsilon, 0.01)

    if np.random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
    return action


def store(state, action, reward, done, next_state):
    buffer.append([state, action, reward, done, next_state])
    return len(buffer)


def fit():
    sample_batch = random.sample(buffer, batch_size)
    for state, action, reward, done, next_state in sample_batch:
        with tf.GradientTape() as tape:
            target = reward if done else reward + gamma * model(np.expand_dims(next_state, axis=0)).numpy().max()
            y_true = model(np.expand_dims(state, axis=0)).numpy()
            y_true[0][action] = target
            loss = tf.keras.losses.mse(y_true,  model(np.expand_dims(state, axis=0)))
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))


i_episode = 0
model = build_model()

while True:

    i_episode += 1
    step_time = 0

    state = env.reset()
    
    while True:
        step_time += 1
        env.render()

        action = select_action(state)

        next_state, reward, done, _ = env.step(action)

        if store(state, action, reward, done, next_state) >= batch_size:
            fit()

        state = next_state

        if done:
            break

    print("i_episode：{} \t step_time：{}".format(i_episode, step_time))
