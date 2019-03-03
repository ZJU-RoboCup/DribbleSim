# -*- coding: utf-8 -*-

import gym
import gym_dribble

env = gym.make('Dribble-v0')

obs = env.reset()
for _ in range(500):
    for _ in range(10):
        action = 0.02
        new_obs, reward, done, _ = env.step(action)
        #print(new_obs, reward, done)
        #if done:
        #    break
    obs = env.reset()
    
env.close()