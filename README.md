# gym-mediator
Decision logic simulation of Mediator system. 

## Introduction
Five use cases are defined in Mediator system, and the decision logic agent is expected to determine the optimal actions to guarantee driving safety and driver comfort. In this project, a computer simulation with existing reinforcement learning algorithms, including implementation and evaluation is conducted for each use case. The research tasks mainly include:

+ To build a specific Markov Decision Process (MDP) model for the decision making problem of each use case, *i.e.*, to determine the state space, action space, transition probability, and reward functions, *etc*.
+ To implement existing reinforcement learning algorithm and evaluate the performance with the baseline decision tree-based policy in terms of driving safety, driver comfort, and time-efficiency, *etc*.



## Requirements

### **OpenAI Baselines**

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms. 
We would implement reinforcement learning algorithms from baselines in our simulator and you have to install baselines first.
For details, see https://github.com/openai/baselines.


### **Other Libaries**
+ scipy
+ matplotlib
+ numpy
+ typing
+ pyglet
+ dataclasses
+ enum
+ times


## Simulation Environment
* For more information about creating a new gym-based environment, see: 
 https://github.com/openai/gym/edit/master/docs/creating-environments.md
 
* In this project, we create a new repo called **gym-mediator**, which should be a pip package and include the following files:

  ```sh
  gym-mediator/
    README.md
    setup.py
    dataset
    algorithms
    figures
    gym_mediator/
      __init__.py
      envs/
        __init__.py
        mediator_env.py
        common
  ```

* The environment is defined in 'mediator_env.py', including:

  ```python

  class MediatorEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
      ...
    def step(self, action):
      ...
    def reset(self):
      ...
    def render(self, mode='human'):
      ...
    def close(self):
      ...
  ```

## Installation

Notes: gym_mediator is included in gym-mediator

$ git clone git@gitlab.ewi.tudelft.nl:yangli_algo/gym-mediator.git

$ cd gym-mediator
  
$ pip install -e .



## Usage
To test the installation of the simulation environment:

$ python algorithms/env_demo.py



## Baselines Reinforcement Learning Algorithms

To test the baseline reinforcement algorithm:

$ python baselines/baselines/deepq/experiments/train_mediator.py 



## Documentation

TBD(see blogs)


 
