
import numpy as np
from scipy.stats import bernoulli, multinomial
import copy
from gym_mediator.envs.common.utils import ProviderState, DriverState, AutomationState, DiscreteAction
from gym_mediator.envs.common.scenario_setup import ScenarioCreate, ScenarioModel
from gym_mediator.envs.common.utils import ScenarioCategory



class MEDIATOR:
    
    def __init__(
        self,
        time_step,
        scenario,
    ):
              
        self._timestep  = time_step       
        self._scenario  = scenario
        self._is_setup  = False   

        self._scenario_model: ScenarioModel = None   
        self._state: ProviderState = None
        self._last_state: ProviderState = None

        # transition probabiliy distribution
        self._distraction_tp_p1 = 0.5   
        self.response_prob = [0.1, 0.8, 0.1]
        
      

    def step(self, agent_action):
        
        '''
        Return: next observations, reward, done, {} 
        '''

        if not self._is_setup:      
            raise MediatorNotSetupError("Must call reset() before stepping.")  
        
        ## 0. agent action,translate to specific actions, e.g., Emergent stop
        action = DiscreteAction(agent_action)

        ## 1. step to next state s_{t+1} based on current state s_{t} and a_{t}
        
        self._state, reached_max_eps = self._scenario_model.step(self._state, action)
        if reached_max_eps:
            print('Episode ends because of maximum length')        
        
        ## 2. agent_done and environmental reward
        env_reward     = 0.0      # need to be customized
        reached_goal   = False
        risk_threshold = 5        # threshold of time elapsed since critical begins

        if ScenarioCategory(self._scenario) == ScenarioCategory.driver_unfit_distraction:
            # for corrective-distraction action case
            distract_last  = self._last_state.driver_state[0].Distraction == 1
            distract_now   = self._state.driver_state[0].Distraction == 1
            cd_activated   = self._state.fb_state[0].Action_Status == 1
            goal_correct   = True if ((distract_last and not distract_now) and cd_activated) else False

            auto_clear     = True if ((distract_last and not distract_now) and not cd_activated) else False
            
            Driver_Unfit_L0 = self._last_state.driver_state[0].Distraction == 1 and self._last_state.automation_state[0].Automation_Mode==0
            Shift_L4     = self._state.automation_state[0].Automation_Mode==3 
            goal_shift      = True if (Driver_Unfit_L0 and Shift_L4) else False



            # reached_goal
            reached_goal = (goal_correct or goal_shift)

        # risk_event
        collision_event = True if self._state.tt_state[0].TESS > risk_threshold else False
        Shift_L4_activated = self._state.fb_state[0].Action_Status == 3  
        risk_action = True if self._state.automation_state[0].LevelMaxNow<3 and Shift_L4_activated else False
        risk_event = collision_event or risk_action
        # emergency_stop
        move_last = self._last_state.context_state[0].Vehicle_Stop==0 
        stop_now  = self._state.context_state[0].Vehicle_Stop==1
        emergency_stop  = True if (move_last and stop_now) else False

        


        ## 3. episode termination
        agent_done   = (
            reached_goal
            or reached_max_eps
            or risk_event
            or emergency_stop
            or auto_clear
        )

        infos_ = {
            'goal': reached_goal,
            'max_eps': reached_max_eps,
            'risk_events': risk_event,
            'emergency_stop': emergency_stop,
            'auto_clear':auto_clear
        }

        ## 4. update last state
        self._last_state = copy.deepcopy(self._state)

        return self._state, env_reward, agent_done, infos_
   
        
  

    def reset(self):

        '''
        Reset the scenario
        '''  
        # To simulate the state sequences: env_obs_all
        scenarios = ScenarioCreate(self._scenario, self._timestep)       

        # To enable dynamics
        self._scenario_model = ScenarioModel(scenarios, self._timestep)

        # Initialization
        self._state  = self._scenario_model.scenario_init()

        # Update last_state
        self._last_state = copy.deepcopy(self._state)

        self._is_setup = True
        

        return self._state
        

        
    def destroy(self):
        
        # Simulation ends and cumulative_sim_time = 0
        self._scenario_model.teardown()
        