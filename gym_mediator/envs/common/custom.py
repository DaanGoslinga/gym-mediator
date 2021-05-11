from dataclasses import dataclass
from typing import Callable
import numpy as np
from gym_mediator.envs.common.utils import ScenarioCategory
from gym_mediator.envs.common.utils import ProviderState, DriverState, AutomationState, DiscreteAction



@dataclass
class AgentSpec:
        
    observation_adapter: Callable = lambda obs: obs
    reward_adapter: Callable = lambda last_obs, action, cur_obs, next_obs, case_id: reward



def observation_custom(env_obs):
    '''
    About the observation design
    Transform the environment's observation into something more suited for your model 
    '''
    def dict_to_array(obs_dict):
        array = []
        for key in obs_dict:
            array = [*array, *obs_dict[key]]
        return np.array(array)

    # # TODO: full states
    obs = {
        "Distraction": np.array([env_obs.driver_state[0].Distraction]),
        "Fatigue": np.array([env_obs.driver_state[0].Fatigue]),
        "Comfort": np.array([env_obs.driver_state[0].Comfort]),
        "Personal_Preference": np.array([env_obs.driver_state[0].Personal_Preference]),
        "AutomationMode": np.array([env_obs.automation_state[0].Automation_Mode]),
        "LevelMaxNow": np.array([env_obs.automation_state[0].LevelMaxNow]),
        "LevelMaxNext": np.array([env_obs.automation_state[0].LevelMaxNext]),
        "TTDF": np.array([env_obs.tt_state[0].TTDF]),
        "TTDU": np.array([env_obs.tt_state[0].TTDU]),
        "TTAF": np.array([env_obs.tt_state[0].TTAF]),
        "TTAU": np.array([env_obs.tt_state[0].TTAU]),
        "TTDD": np.array([env_obs.tt_state[0].TTDD]),
        "TESD": np.array([env_obs.tt_state[0].TESD]),
        "TESS": np.array([env_obs.tt_state[0].TESS]),
        "SC": np.array([env_obs.context_state[0].Scenario_Critical]),
        "UC": np.array([env_obs.context_state[0].Uncomfort_Event]),
        "VS": np.array([env_obs.context_state[0].Vehicle_Stop]),
        "TS": np.array([env_obs.context_state[0].Time_Stamp]),
        "Driver_Choice": np.array([env_obs.fb_state[0].Driver_Choice]),
        "Action_Status": np.array([env_obs.fb_state[0].Action_Status]),
    }

    # # TODO: to customize the states

    # obs = {
    #     "DistractionLast": np.array([last_obs.driver_state[0].Distraction]),
    #     "DistractionNow": np.array([env_obs.driver_state[0].Distraction]),
    #     "LevelMaxNow": np.array([env_obs.automation_state[0].LevelMaxNow]),
    #     "AutomationMode": np.array([env_obs.automation_state[0].Automation_Mode]),
    #     "Driver_Choice": np.array([env_obs.fb_state[0].Driver_Choice]),
    #     "Action_Status": np.array([env_obs.fb_state[0].Action_Status]),
    # }


    obs = dict_to_array(obs)
    obs_custom = obs.reshape((-1,len(obs)))
 

    return obs_custom



def reward_custom(last_env_obs, action, env_obs, next_env_obs, scenario_id):
    '''
    Reward Design
    inputs:
    s_{t-1}, s_{t}, a_{t}, s_{t+1}, case_id
    outputs:
    reward_fun
    '''

    action = DiscreteAction(action)
  
    # last_obs, time t-1
    last_d_att = int(last_env_obs.driver_state[0].Distraction)
    
    # current observation, time t
    auto_mode  = int(env_obs.automation_state[0].Automation_Mode)
    L_max_now  = int(env_obs.automation_state[0].LevelMaxNow)
    L_max_next  = int(env_obs.automation_state[0].LevelMaxNext)

    d_att  = int(env_obs.driver_state[0].Distraction)
    f_as   = int(env_obs.fb_state[0].Action_Status)   
    f_dc   = int(env_obs.fb_state[0].Driver_Choice)   

    next_d_att = int(next_env_obs.driver_state[0].Distraction)
    next_f_as   = int(next_env_obs.fb_state[0].Action_Status)   

    L4_available = L_max_now ==3 and L_max_next==3
    cd_activated = True if f_as ==1 else False

    ssl4_activated = f_as ==2
    #######################################
    '''
    Penalty + Reward for Distraction-induced degraded driver behavior
    '''
    reward_all = 0.0
    costs,rewards = 0.0, 0.0

    # Cost 1: degraded driver performance
    if d_att==1 and next_d_att==1:
        costs = costs - 0.5
    if (d_att==1 and next_d_att==0) and action == DiscreteAction.Correct_Distraction:
        rewards = rewards + 5     # rewards when reaching targets

    # Branch 
    '''[d_att, cd_activated,L4_available]: four conditions
    0,0
    0,1
    1,0
    1,1
    '''
    
    if d_att == 1 and (not cd_activated and not L4_available):
        if action == DiscreteAction.Correct_Distraction:
            rewards  = rewards + 2.0  # *
        elif action == DiscreteAction.Suggested_Shift_L4:
            costs = costs - 2.0      # reduce driver comfort
        elif action == DiscreteAction.Shift_L4:
            costs = costs - 10.0      # increase risk significantly               

    if d_att == 1 and (not cd_activated and L4_available):
        if action == DiscreteAction.Correct_Distraction:
            rewards  = rewards + 2.0  # *
        elif action == DiscreteAction.Suggested_Shift_L4:
            rewards  = rewards + 0.1
        elif action == DiscreteAction.Shift_L4:
            costs = costs - 5.0      # reduce driver comfort

    if d_att == 1 and (cd_activated and not L4_available):
        if action == DiscreteAction.Correct_Distraction:
            rewards  = rewards + 2.0 # *
        elif action == DiscreteAction.Suggested_Shift_L4:
            costs = costs - 2.0      # increase risk
        elif action == DiscreteAction.Shift_L4:
            costs = costs - 10.0      # increase risk significantly          




    if d_att == 1 and (cd_activated and L4_available):
        '''
        ssl4_activated, driver_response
        0, -1
        1, 0
        1, 1
        1, 2
        '''

        if not ssl4_activated:
            if action == DiscreteAction.Suggested_Shift_L4:
                rewards  = rewards + 2.0       # * 
            elif action == DiscreteAction.Shift_L4:
                costs = costs - 5.0            # reduce driver comfort
            elif action == DiscreteAction.Correct_Distraction:
                rewards = rewards + 0.5        # 

        if ssl4_activated and f_dc ==0:
            # no response
            if action == DiscreteAction.Suggested_Shift_L4:
                rewards  = rewards + 2.0      # *
            elif action == DiscreteAction.Shift_L4:
                costs = costs - 10.0          # [high]reduce driver comfort
            elif action == DiscreteAction.Correct_Distraction:
                costs = costs - 2             # reduce driver comfort        


        if ssl4_activated and f_dc ==1:
            # accept
            if action == DiscreteAction.Suggested_Shift_L4:
                costs   = costs - 2.0        # reduce driver comfort  
            elif action == DiscreteAction.Shift_L4:
                rewards = rewards + 2.0       # *
            elif action == DiscreteAction.Correct_Distraction:
                costs = costs - 2.0          # reduce driver comfort 


        if ssl4_activated and f_dc ==2:
            # reject
            if action == DiscreteAction.Correct_Distraction:
                rewards = rewards + 2.0    # * 
            elif action == DiscreteAction.Suggested_Shift_L4:
                costs = costs - 2.0        # reduce driver comfort
            elif action == DiscreteAction.Shift_L4:
                costs = costs - 10.0       # [high]reduce driver comfort 
                

    # sum all
    reward_all = rewards + costs 
    # if d_att==1:
    #     print('>> Rewards = {}, Costs = {}, Rewards+Costs = {}'.format(rewards,costs,reward_all))

    return round(reward_all,1)
















