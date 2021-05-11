# This script is used to test the installation of the environment.

import gym
import gym_mediator
import random 

env = gym.make("mediator-v0")
env.reset()
print('Well done, Mediator simulator is installed successfully!')

count = 0
while count<1:
    # Randomly get an action
    action = random.randint(0,3)
    obs, reward, done, infos = env.step(action)
    count = count + 1
    print('simulation infos',infos)
    
