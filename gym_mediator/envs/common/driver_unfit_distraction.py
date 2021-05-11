from gym import spaces
import numpy as np
import random
import numpy as np
from gym_mediator.envs.common.distraction import DistractionSequence
from gym_mediator.envs.common.TTA_sim import compute_ttax
from gym_mediator.envs.common.TTD_sim import compute_ttdx
from scipy.stats import bernoulli, multinomial
from enum import Enum
from typing import List
from dataclasses import dataclass, field



UNCOMFORT_EVENT = {
    0: 'None',
    1: 'Traffic_Jam', 
    2: 'Poor_Visibility', 
    3: 'Long_Trips', 
    4: 'Motion_Sickness', 
    5: 'Time_Pressure', 
}

MODE = {
    0: 'L0',
    1: 'L2', 
    2: 'L3',
    3: 'L4',
}

NDRT = {
    0: "None", 
    1: 'Messaging',
    2: 'Obstruction',
    3: 'Immersion', 
    4: 'Obstruction_Immersion',
}

DISTRACTION = {
    0: 'Sufficient SA',
    1: 'Long term loss of SA',
}

FATIGUE = {
    0: 'Alert',
    1: 'Neither alert nor sleepy',
    2: 'Some signs of sleepiness',
    3: 'Sleepy',
}

DRIVER_REQUEST = { 
    0: "None",
    1: "Shift to L0",
    2: "Shift to L2",
    3: "Shift to L3", 
    4: "Shift to L4",
}

DRIVER_RESPONSE = {
    0: "No Response", 
    1: "Accept",
    2: "Reject",
}


''' TODO: Define the state providers as follows.
    + DriverState
    + AutomationState
    + TimeMetricsState
    + ContextState
    + FeedBackState
'''
@dataclass
class DriverState:
    Distraction: int    
    Fatigue: int 

@dataclass
class AutomationState:
    Automation_Mode: int
    Level_Max: int

@dataclass
class TimeMetricsState:
    TTDF: float 
    TTDU: float 
    TTDD: float 
    TTA2F: float
    TTA3F: float
    TTA4F: float
    TTA2U: float 
    TTA3U: float 
    TTA4U: float 

@dataclass
class ContextState:
    NDRT_Type: int
    Uncomfort_Event: int
    Sensor_Failure: int
    Leave_ODD: int

@dataclass
class FeedBackState:
    Driver_Response: int
    Driver_Request: int
    Last_Action: int
    Vehicle_Stop: int


@dataclass
class ProviderState: 
    # to summarize all states
    driver_state: DriverState        
    automation_state: AutomationState 
    tt_state: TimeMetricsState        
    context_state: ContextState       
    fb_state: FeedBackState         




class DriverUnfitDistraction(object):

    def __init__(self):

        self.actions = {
            0: 'Do_Nothing',
            1: 'Correct_Distraction',
            2: 'Shift_Lopt_Suggest',
            3: 'Shift_Lopt_Enforced',
            4: 'Emergency_Stop',
        }

        self._configs = self.default_config()
        
    @classmethod
    def default_config(cls) -> dict:
        """
        TODO: define your model parameters
        -----------------------------------
        + np_uc: prediction horizon for uncomfortable events
        + response_probability: probability of driver's response to the suggestion
        + TTX_ub: upper bound of time metrics
        """
        return {
            "time_step": 1,
            "np_uc": 300,
            "auto_mode": "L0",
            "distraction": True,
            "fatigue": False,
            "ndrt": "None",
            "uc_event": "None",
            "sensor_failure": False,
            "driver_initiate_request": False,
            "distraction_tp": 0.5,
            "response_probability": [0.1, 0.8, 0.1],
            "TTX_ub": 9999,
        }
   
    
    def reset_scenario_lookup(self):    
        ''' 
        Create the lookup table: simulations for the variables defined below
        + distraction_level 
        + fatigue_level

        + auto_level
        + level_max  
        + leave_ODD(TODO)        
        + sensor_failure

        + ndrt_type
        + uc_event

        + TTDF
        + TTDU
        + TTDD
        + TTAF: TTA2F/TTA3F/TTA4F
        + TTAU: TTA2U/TTA3U/TTA4U

        + driver_request
        + driver_response
        + last_action
        + vehicle_stop
        '''

        # Driver
        dt = self._configs["time_step"]

        # TODO: distraction
        if self._configs["distraction"]:
            print('Distraction file is loaded')
            distraction_file  = DistractionSequence(dt).load()  # [time,time_buffer,distracted]
            self.seq_len = len(distraction_file) + 80           #  len(distraction_file) around 40s.       
            distraction_start = min(np.where(np.array(distraction_file)[:,2]>0)[0])  
            distraction_level = np.array(distraction_file)[:,2]
        else:
            print('Distraction not found')

        fatigue_level = np.ones((self.seq_len,1)) * int(self._configs["fatigue"])

        # Automation 
        temp_mode  = list(MODE.keys())[list(MODE.values()).index(self._configs["auto_mode"])]
        high_mode  = list(MODE.keys())[list(MODE.values()).index("L4")]
        auto_level = temp_mode * ones((self.seq_len,1))
        # TODO: level_max
        level_max  = random.randint(temp_mode,high_mode+1) * np.ones((self.seq_len,1))

        # Context
        sensor_failure = int(self._configs["sensor_failure"]) * np.ones((self.seq_len,1))
        ndrt_type = list(NDRT.keys())[list(NDRT.values()).index(self._configs["ndrt"])] * np.ones((self.seq_len,1))
        uc_event = list(UNCOMFORT_EVENT.keys())[list(UNCOMFORT_EVENT.values()).index(self._configs["uc_event"])] * np.ones((self.seq_len,1))
        # TODO: leave_odd

        
        # Time Metrics
        TTDF = np.nan * np.ones((self.seq_len,1))
        TTDU = self._configs["TTX_ub"] * np.ones((self.seq_len,1))
        TTDD = self._configs["TTX_ub"] * np.ones((self.seq_len,1))
        TTA2F,TTA3F,TTA4F,TTA2U,TTA3U,TTA4U = compute_ttax(self.seq_len,auto_level,level_max, dt)

        # HMI
        driver_response =  0 * np.ones((self.seq_len,1))
        driver_request  = int(self._configs["driver_initiate_request"]) * np.ones((self.seq_len,1))
        
        # Other
        last_action  = list(self.actions.keys())[list(self.actions.values()).index("Do_Nothing")] * ones((self.seq_len,1))
        vehicle_stop = 0 * np.ones((self.seq_len,1)) # default: 0 (vehicle is moving)

        # summary of all states
        self.human_obs   = np.hstack((distraction_level,fatigue_level))
        self.auto_obs    = np.hstack((auto_level,level_max))
        self.tt_obs      = np.hstack((TTDF,TTDU,TTDD,TTA2F,TTA3F,TTA4F,TTA2U,TTA3U,TTA4U))
        self.context_obs = np.hstack((ndrt_type,uc_event,sensor_failure,leave_odd))
        self.fb_obs      = np.hstack((driver_response,driver_request,last_action,vehicle_stop)))


    def sim_to_provider(self):
        '''Environmental Observations >> States Providers '''
        human_provider = DriverState(                    
            Distraction = self.distraction_value, 
            Fatigue     = self.fatigue_value,
        )
                
        auto_provider = AutomationState(
            Automation_Mode   = self.auto_level_value,
            Level_Max         = self.level_max_value, 
        )   

        ttx_provider = TimeMetricsState(
            TTDF = self.TTDF_value,
            TTDU = self.TTDU_value,
            TTDD = self.TTDD_value,
            TTA2F = self.TTA2F_value,
            TTA3F = self.TTA3F_value,
            TTA4F = self.TTA4F_value,
            TTA2U = self.TTA2U_value,
            TTA3U = self.TTA3U_value,
            TTA4U = self.TTA4U_value,                              
        )

        context_provider = ContextState(
            NDRT_Type         = self.ndrt_type_value,
            Uncomfort_Event   = self.uc_event_value,
            Sensor_Failure    = self.sensor_failure_value,
            Leave_ODD         = self.leave_odd_value,
        )

        feedback_provider = FeedBackState(
            Driver_Response   = self.driver_response_value,
            Driver_Request    = self.driver_request_value,
            Last_Action       = self.last_action_value,
            Vehicle_Stop      = self.vehicle_stop_value,
        )

        return ProviderState(            
            driver_state     = human_provider,
            automation_state = auto_provider,   
            tt_state         = ttx_provider, 
            context_state    = context_provider,
            fb_state         = feedback_provider,
        )



    def _to_space(self) -> spaces.Space:
        '''Return observation_space, action_space'''
        # TODO: define your state space of MDP
        # self._obs_dict: value of the dictionary is the [lower, upper] bound of the variable
        self._obs_dict = {
            "distraction": np.array([0,1]),
            'auto_level': np.array([0,3]),
            'level_max': np.array([0,3]),
            'ttdu': np.array([0,self._configs["TTX_ub"]]),
            'ttdd': np.array([0,self._configs["TTX_ub"]]),
            'tta2f': np.array([0,self._configs["TTX_ub"]]),
            'tta3f': np.array([0,self._configs["TTX_ub"]]),
            'tta4f': np.array([0,self._configs["TTX_ub"]]),
            'tta2u': np.array([0,self._configs["TTX_ub"]]),
            'tta3u': np.array([0,self._configs["TTX_ub"]]),
            'tta4u': np.array([0,self._configs["TTX_ub"]]),
            'uc_event': np.array([0,5]),
            'leave_odd': np.array([0,1]),
            'driver_response': np.array([0,2]),
            'last_action': np.array([0,len(self.actions)-1]),
            'vehicle_stop': np.array([0,1])      
        }
        
        lb_array, ub_array = np.zeros(len(self._obs_dict)), np.zeros(len(self._obs_dict))
        obs_id = 0
        for key, value in self._obs_dict.items():
            lb_array[obs_id], ub_array[obs_id] = value[0], value[1]
            obs_id+=1
        
        observation_space = spaces.Box(low=lb_array, high=ub_array, dtype=np.float32)
        action_space = spaces.Discrete(len(self.actions))
        
        return observation_space, action_space




    def update_obs_lookup(self, index):
        ''' Update the observations based on the >> Lookup Table 
        '''
        self.distraction_value = int(self.human_obs[index,0])
        self.fatigue_value     = int(self.human_obs[index,1])

        self.auto_level_value = int(self.auto_obs[index,0]) 
        self.sensor_failure_value = int(self.context_obs[index,2])
        self.leave_odd_value      = int(self.context_obs[index,3])  
        if self.sensor_failure_value:
            self.level_max_value = 0
        else:
            self.level_max_value = int(self.auto_obs[index,1]) 

        self.ndrt_type_value = int(self.context_obs[index,0])
        self.uc_event_value  = int(self.context_obs[index,1])

        self.TTDF_value = self.tt_obs[index,0]
        self.TTDU_value = self.tt_obs[index,1]

        if self.distraction_value ==1:
            self.TTDU_value = self.tt_obs[index,1]





        self.TTDD_value = self.tt_obs[index,2]
        self.TTA2F_value = self.tt_obs[index,3]
        self.TTA3F_value = self.tt_obs[index,4】
        self.TTA4F_value = self.tt_obs[index,5]
        self.TTA2U_value = self.tt_obs[index,6]
        self.TTA3U_value = self.tt_obs[index,7]
        self.TTA4U_value = self.tt_obs[index,8]

        self.driver_response_value = int(self.fb_obs[index,0])
        self.driver_request_value  = int(self.fb_obs[index,1])
        self.last_action_value     = int(self.fb_obs[index,2])
        self.vehicle_stop_value    = int(self.fb_obs[index,3])

        if self.distraction_value==1:
            self.distraction_starts = True

   
    def update_obs_act(self, action):
        ''' Update the observations based on >> Transition Functions
        '''
        obs_now = self.states_provider

        self.last_action_value = action 

        if self.actions[action]== "Do_Nothing":
            print('action:Do_Nothing')

        elif self.actions[action]=="Correct_Distraction":
            if obs_now.driver_state.Distraction == 1:
                # sampled from a bernoulli distribution p(x=1) = self._configs["distraction_tp"]
                self.distraction_value = bernoulli.rvs(size=1, p=self._configs["distraction_tp"])[0]
                if self.distraction_value == 0: 
                    # TODO: 
                    self.TTDF_value = 0.0
                    self.TTDU_value = self._configs["TTX_ub"]
                    self.correction_distraction = True
                

        elif self.actions[action]=="Shift_Lopt_Suggest":
            driver_response_distribution  = multinomial.rvs(1,np.array(self._configs["response_probability"]),size=1)
            self.driver_response_value = np.where(driver_response_distribution[0]==1)[0][0]
        

        elif self.actions[action]=="Shift_Lopt_Enforced":
            level_optimal = self.level_max_value
            if level_optimal>obs_now.automation_state.Automation_Mode:
                self.auto_level_value = level_optimal
                self.shift_done = True
            else:
                print('Shfit Failed!)


        elif self.actions[action]=="Emergency_Stop":
            self.vehicle_stop_value = 1
            self.emergency_stop = True



    def _to_reset(self):
        '''Reset Environment >> self.state '''
        self.step_id = 0

        # Reset lookup table
        self.reset_scenario_lookup()
        self.update_obs_lookup(self.step_id)

        # Initialize flags
        self.distraction_starts = False
        self._done = False
        self.reach_goal = False
        self.correction_distraction = False
        self.shift_done = False
        self.reach_max_episode = False
        self.emergency_stop = False


        # Initialize states
        self.states_provider = self.sim_to_provider()

        return self.customize_observation(self.states_provider)


    def _to_step(self, action: int):
        '''
        Step the environment
        =========================
        Return: s, r, done, infos
        '''

        self.step_id +=1

        # record the current states
        self._current_states_provider = self.states_provider

        # Update state
        ## first, update observations based on the lookup table
        ## next, update observations based on transition functions
        ## then, turn observations to states

        self.update_obs_lookup(self.step_id)
        self.update_obs_act(action)
        self.states_provider = self.sim_to_provider() 
        reward = self.reward_design(self._current_states_provider, action, self.states_provider)
        self._done, infos = self.agent_is_done()
        
        return self.customize_observation(self.states_provider), reward, self._done, infos


    def agent_is_done(self):
        '''To determine whether the agent is done or not'''

        # result 1: reached_goal: <Distraction Corrected> or <Shift Done>
        self.reach_goal = self.correction_distraction or self.shift_done

        # result 2: reach_max_episode(i.e., seq_len)
        if self.step_id==self.seq_len:
            self.reach_max_episode = True
        # Done
        agent_done   = (
            self.reach_goal
            or self.reach_max_episode
            or self.emergency_stop
        )
        # Infos
        infos = {
            'reach_goal': self.reach_goal,
            'reach_max_episode': self.reach_max_episode,
            'emergency_stop': self.emergency_stop
        }        

        return agent_done, infos


    def reward_design(self, current_state, action, next_state):
        ''' Customize your reward here: R(s,a,s') '''
        # TODO

        return env_reward


    def customize_observation(self, state_provider):
        ''' Customize your observations >> obs_dict ''' 
        # TODO: modify obs_dict based on your MDP model
        
        self._obs_dict.update({
            "distraction": np.array([state_provider.driver_state.Distraction]),
            'auto_level': np.array([state_provider.automation_state.Automation_Mode]),
            'level_max': np.array([state_provider.automation_state.Level_Max]),
            'ttdu': np.array([state_provider.tt_state.TTDU]),
            'ttdd': np.array([state_provider.tt_state.TTDD]),
            'tta2f': np.array([state_provider.tt_state.TTA2F]),
            'tta3f': np.array([state_provider.tt_state.TTA3F]),
            'tta4f': np.array([state_provider.tt_state.TTA4F]),
            'tta2u': np.array([state_provider.tt_state.TTA2U]),
            'tta3u': np.array([state_provider.tt_state.TTA3U]),
            'tta4u': np.array([state_provider.tt_state.TTA4U]),
            'uc_event': np.array([state_provider.context_state.Uncomfort_Event]),
            'leave_odd': np.array([state_provider.context_state.Leave_ODD]),
            'driver_response': np.array([state_provider.fb_state.Driver_Response]),
            'last_action':np.array([state_provider.fb_state.Last_Action]),
            'vehicle_stop': np.array([state_provider.fb_state.Vehicle_Stop])   
        })

        obs_temp = self.dict_to_array(self._obs_dict)
        
        return obs_temp.reshape((-1,len(obs_temp)))


    def dict_to_array(self, obs_dict):
        array = []
        for key in obs_dict:
            array = [*array, *obs_dict[key]]
        return np.array(array)




def scenario_factory(config):
    '''Choose simulation cases '''
    if config["casetype"] == "driver_unfit_distraction":
        print('>> Scenario Initialization: driver_unfit_distraction <<')
        return DriverUnfitDistraction()
    else:
        raise ValueError("Unknown case type")













































