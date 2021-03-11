import os
import gym
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = gym.make("Pendulum-v0")

print("#####################TEST###################")
print("observation_space shape: {}".format(env.observation_space.shape))
print("observation_space low: {}".format(env.observation_space.low))
print("observation_space high: {}".format(env.observation_space.high))
print("action_space shape: {}".format(env.action_space.shape))
print("action_space low: {}".format(env.action_space.low))
print("action_space high: {}".format(env.action_space.high))     
print("reward_range: {}".format(env.reward_range))
print("########################################")


for i_episode in range(20):
    observation = env.reset()

    for t in range(100):

        # plt.imshow(observation)
        # plt.show()

        env.render()
         
        action = env.action_space.sample()
        # print(action)

        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
