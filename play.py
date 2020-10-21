import gym

from stable_baselines3 import PPO
from stable_baselines3 import DDPG
ENV_NAME = 'Pendulum-v0'
#ENV_NAME = 'CartPole-v1'
env = gym.make(ENV_NAME)

#model = PPO('MlpPolicy', env, verbose=1)
model = DDPG('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()