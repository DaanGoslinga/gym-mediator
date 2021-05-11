
import gym
from gym.envs.registration import register
from gym import spaces, logger
from gym.utils import seeding
from gym_mediator.envs.common.driver_unfit_distraction import scenario_factory


# define the gym-based environment
CASE_LIB = {
    0: "driver_unfit_distraction",
    1: "driver_unfit_fatigue",
    2: "automation_unfit",
    3: "driver_initiate_shift",
    4: "mediator_initiate_shift",
}

class MediatorNewEnv(gym.Env):

    def __init__(self):
        
        # TODO: check case_id >> CASE_LIB 
        case_id = 0
        config_info = {
            "casetype": CASE_LIB[case_id],
        }

        mediator_sim = scenario_factory(config_info)
        self.observation_space, self.action_space = mediator_sim._to_space()

    def step(self,action):
        return mediator_sim._to_step(action)

    def reset(self):
        return mediator_sim._to_reset()

    

# Register the environment

register(
    id='mediator-v1',
    entry_point='gym_mediator.envs:MediatorNewEnv',
    max_episode_steps=200,
)

