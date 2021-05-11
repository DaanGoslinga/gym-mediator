import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import bernoulli, multinomial

from gym_mediator.envs.common.distraction import DistractionSequence
from gym_mediator.envs.common.utils import ScenarioCategory
from gym_mediator.envs.common.utils import DriverState, AutomationState, TimeMetricsState, FeedBackState, ContextState
from gym_mediator.envs.common.utils import ProviderState, DiscreteAction
from gym_mediator.envs.common.utils import UncomfortableEvent




class ScenarioCreate:   
    
    def __init__(
        self,
        scenario_type:int,
        step_size, 
    ):
        
        self._scenario             = ScenarioCategory(scenario_type)
        self._sim_step             = step_size
        self.probab_driver_resp    = 0.95
        self.probab_critical_case  = 0.1
    #     self.seed()
    
    # def seed(self, seed = None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
        
        
    def generate_sequence(self):
        

        '''
        Driver degraded-distraction
        '''        
        if self._scenario == ScenarioCategory.driver_unfit_distraction:
                     
           ##----------------------------- Driver State ---------------------------------##      
            # print('>> Initialization: Driver Distraction-related Degraded Scenario')
            
            _distraction_segment = DistractionSequence(self._sim_step).load() 

            time_seg             = np.array(_distraction_segment)[:,0]  # time frame
            distraction_slot     = np.where(np.array(_distraction_segment)[:,2]>0)[0]  # the moment when distraction occurs
            seq_len              = len(_distraction_segment)            # maximum length of the generated scenario

            _fatigue_segment   = np.zeros((seq_len,1))  # not fatigue, set 0
            _comfort_segment   = np.ones((seq_len,1))   # comfortable, set 1

            # driver's preference, 0 - no preference; 1 - prefer manual driving; 2-prefer automated driving
            _personal_preference = np.zeros((seq_len,1))  # no preference, set 0

            # TODO: driver distraction: 
            _distraction_segment = np.array(_distraction_segment) # 3 columns
            _distraction_segment[distraction_slot[0]:seq_len,2] = np.ones((seq_len-distraction_slot[0]))

            driver_seg          = np.hstack((_distraction_segment, _fatigue_segment, _comfort_segment, _personal_preference))
            # print('seq_len: {}, distraction begin at: {}, '.format(seq_len,distraction_slot[0]))


            ## ------------------------ Automation State -------------------------------##
            # -AutomationMode: int, 0-M, 1-CM, 2-L3, 3-L4
            # -LevelMaxNow: int 
            # -LevelMaxNext: int
            # -AutomationPerformance: int, Reliable(1), Not Reliable(0)
        
            current_level  = 0     #random.randint(0,2), random selection from L0/l2/L3
            _Auto_Mode     = current_level * np.ones((seq_len,1))  
            _LevelMaxNow   = random.randint(current_level,3) * np.ones((seq_len,1))
            _LevelMaxNext  = random.randint(current_level,3) * np.ones((seq_len,1))              
            auto_seg       = np.hstack((_Auto_Mode, _LevelMaxNow, _LevelMaxNext))


            ##------------------------------ Context Observations-------------------------------------##

            # Scenario criticality
            # Uncomfortable event
            # Vehicle stop
            _scenario_critical = np.zeros((seq_len,1))
            # Time elapsed since critical begins
            _TESS = np.zeros((seq_len,1))
            
            if bernoulli.rvs(size=1, p = self.probab_critical_case):  # the probability of critical scenario is 0.05
                # TODO: extreme cases

                sc_begin = random.randint(round(seq_len*0.2),round(seq_len*0.8))                    # critical begins
                sc_end   = sc_begin + 30      # critical ends
                if sc_end>seq_len-1:
                    sc_end = seq_len-1
                
                _scenario_critical[sc_begin:sc_end,:] = np.ones((sc_end - sc_begin, 1))
                _TESS[sc_begin:sc_end,:] = np.linspace(0, self._sim_step*(sc_end - sc_begin - 1), sc_end - sc_begin)[:, np.newaxis]
            
            _uncomfortable_event = np.zeros((seq_len,1))
            _vehicle_stop        = np.zeros((seq_len,1))  
            _time_stamp          = np.zeros((seq_len,1)) 

            context_seg = np.hstack((_scenario_critical, _uncomfortable_event, _vehicle_stop, _time_stamp)) 



            ##-------------------------- Time Metrics ----------------------------------##
            # _TTDF and _TTDU
            # _TTAF and _TTAU
            # TTDD how to compute?

            _TTDF = np.zeros((seq_len,1))
            _TTDF[distraction_slot[0]:distraction_slot[-1]+1,:] = 300 * np.ones((distraction_slot[-1]-distraction_slot[0]+1,1))

            _TTDU = np.resize(np.linspace(7200, 7200-self._sim_step*(seq_len-1), seq_len),(seq_len,1))  #  TTDU would be at least 2 hour left 
            _TTDU[distraction_slot[0]:distraction_slot[-1]+1,:] = np.zeros((distraction_slot[-1]-distraction_slot[0]+1,1))

            _TTAF = np.zeros((seq_len,1)) # with L2 on
            _TTAU = 7200*np.ones((seq_len,1)) # with L2 works well

            TTDD_0 = np.random.randint(5,120)*60
            _TTDD_continue_driving = np.resize(np.linspace(TTDD_0, TTDD_0 -self. _sim_step*(seq_len-1), seq_len),(seq_len,1))

            TTDD_0 = 0.0
            _TTDD_distraction = np.resize(np.linspace(TTDD_0, TTDD_0 - self._sim_step*(seq_len-1), seq_len),(seq_len,1))

            
            _TTDD = np.zeros((seq_len,1))

            for i in range(seq_len):

                if i < distraction_slot[0]+1:
                    _TTDD[i] = _TTDD_continue_driving[i]
                    
                elif i < distraction_slot[-1]+1:
                    _TTDD[i] = _TTDD_distraction[i]

                else:
                    _TTDD[i] = _TTDD_continue_driving[i]
            
            
            _TESD = np.zeros((seq_len,1))  # Time elapsed since degraded behavior starts
            
            if distraction_slot[0]!= distraction_slot[-1]:
                dd_begin, dd_end = distraction_slot[0], distraction_slot[-1]
                # print('begin={}, end={}'.format(dd_begin, dd_end))
                _TESD[dd_begin:dd_end] = np.linspace(0, self._sim_step *(dd_end - dd_begin - 1), dd_end - dd_begin)[:, np.newaxis]
            
            # elif len(np.where(_fatigue_segment==1)[0])>0:
            #     dd_begin, dd_end = fat_begin, fat_end
            #     _TESD[dd_begin:dd_end] = np.linspace(0, self._sim_step *(dd_end - dd_begin - 1), dd_end - dd_begin)[:, np.newaxis]
            
            # print('_TESD:',_TESD)

            # Stack together
            tt_seg   = np.hstack((_TTDF,_TTDU,_TTAF,_TTAU,_TTDD, _TESD, _TESS))



            ##------------------------------- HMI and Feedbacks ---------------------------------##
            # Driver's response to the suggested action, 0 - no reponse; 1 - accept; 2 - reject
            _driver_choice  = -1*np.ones((seq_len,1))  
            _action_status  = np.zeros((seq_len,1))  
            feedback_seg = np.hstack((_driver_choice, _action_status))



        '''
        Driver degraded-fatigue
        '''

        # elif self._scenario == ScenarioCategory.driver_unfit_fatigue:
            
        #     ##-----------------------------Driver----------------------------------------##
        #     seq_len = 1800 # 3mins
        #     _distraction_segment = np.zeros((seq_len,1))
        #     fatigue_segment     = np.zeros((seq_len,1))
        #     fatigue_flag        = random.randint(50,1200)
        #     fatigue_segment[fatigue_flag:seq_len,:] = np.ones((seq_len-fatigue_flag,1))            


        #     ## -------------------------Automation State-------------------------------##
        #     # -AutomationMode: int, 0-M, 1-CM, 2-L3, 3-L4
        #     # -LevelMaxNow: int 
        #     # -LevelMaxNext: int
        #     # -AutomationPerformance: int, Reliable(1), Not Reliable(0)

        #     current_level = random.randint(0,2) # L0/l2/L3
        #     Mode          = current_level * np.ones((seq_len,1))   
        #     LevelMaxNow   = random.randint(current_level,3) * np.ones((seq_len,1))
        #     LevelMaxNext  = random.randint(current_level,3) * np.ones((seq_len,1))      
        #     AP            = 1 * np.ones((seq_len,1))
        #     auto_seg      = np.hstack((Mode,LevelMaxNow,LevelMaxNext,AP))


        #     ##-----------------------------Time Metrics------------------------------------##
        #     # _TTDF and _TTDU
        #     # _TTAF and _TTAU
        #     _TTDF = np.zeros((seq_len,1))
        #     _TTDF[fatigue_flag:seq_len,:] = 3600*np.ones((seq_len-fatigue_flag,1)) 

        #     _TTDU    = np.ones((seq_len,1))
        #     _TTDU_st = random.randint(2,10)*fatigue_flag * self. _sim_step
        #     _TTDU_ed = _TTDU_st - (fatigue_flag - 1)*self. _sim_step
        #     _TTDU[0:fatigue_flag,:] = np.resize(np.linspace(_TTDU_st, _TTDU_ed, fatigue_flag),(fatigue_flag,1))
        #     _TTDU[fatigue_flag:seq_len,:] = np.zeros((seq_len-fatigue_flag,1))

            
        #     _TTAF = np.zeros((seq_len,1)) # with L2 on
        #     _TTAU = 7200*np.ones((seq_len,1)) # with L2 works well            



        return driver_seg, auto_seg, tt_seg, context_seg, feedback_seg
    
    


class ScenarioModel:
    
    def __init__(
        self,
        scenarios,
        timestep_sec,
    ):
        
        self.human_obs, self.auto_obs, self.tt_obs, self.context_obs, self.fb_obs = scenarios.generate_sequence() 

        self._step       = timestep_sec 
        self._cum_time   = 0.0
        self._timeError  = 0.1
        self.epsicode_ending = False

        self.risk_clear = False

        self.driver_choice = -1
        self.response_prob = [0.1, 0.8, 0.1]

        self.action_flag   = 0
        self.vehicle_stop  = False

        self._distraction_tp_p1 = 0.5   
        self.correction_work = False
        self.turn_L4 = False
        self.episode_time = 0
        

    def scenario_init(self):
        '''
        Return the initial state of the environment
        '''
        # driver state
        init_driver = []
        init_driver.append(           
            DriverState(                    
                Distraction = int(self.human_obs[0,2]), 
                Fatigue     = int(self.human_obs[0,3]),
                Comfort     = int(self.human_obs[0,4]), 
                Personal_Preference = int(self.human_obs[0,5]), 

            )
        )
              
        # automation state        
        init_automation = []
        init_automation.append(          
            AutomationState(
                Automation_Mode         =  int(self.auto_obs[0,0]),
                LevelMaxNow             =  int(self.auto_obs[0,1]),
                LevelMaxNext            =  int(self.auto_obs[0,2]),
  
            )
        )   

        # time metrics
        init_tt = []
        init_tt.append(
            TimeMetricsState(
                TTDF = self.tt_obs[0,0],
                TTDU = self.tt_obs[0,1],
                TTAF = self.tt_obs[0,2],
                TTAU = self.tt_obs[0,3],
                TTDD = self.tt_obs[0,4],
                TESD = self.tt_obs[0,5],
                TESS = self.tt_obs[0,6],
            )
        )

        # context infomation 
        init_context = []
        init_context.append(
            ContextState(
                Scenario_Critical = int(self.context_obs[0,0]),
                Uncomfort_Event   = int(self.context_obs[0,1]),
                Vehicle_Stop      = int(self.context_obs[0,2]),
                Time_Stamp        = int(self.context_obs[0,3]),
            ))


        # action feedback
        init_fb = []
        init_fb.append(
            FeedBackState(
                Driver_Choice   = int(self.fb_obs[0,0]),
                Action_Status   = 0,

            )
        )



        # all states
        state_init = []
        state_init = ProviderState(            
            driver_state     = init_driver,
            automation_state = init_automation,   
            tt_state         = init_tt, 
            context_state    = init_context,
            fb_state         = init_fb,
        )


        return state_init

    


    def step(self,obs_now,action):
        
        '''
        To step state conditioned on the action
        '''

        
        self._cum_time += self._step
        index  = np.where(abs(self.human_obs[:,0]-self._cum_time)<=self._timeError)

        if len(index[0])!=0:

            current_time = self.human_obs[index[0][0],0]
            if current_time + self._step > self.human_obs[-1,0]:
                '''end of episode'''
                self.epsicode_ending = True

            ''' States impacted by the action        
                Define Transitions: P(s'|s,a)
            '''
            if action == DiscreteAction.Do_Nothing:
                self.action_flag = 0

            elif action == DiscreteAction.Correct_Distraction:
                self.action_flag = 1
                if obs_now.driver_state[0].Distraction == 1:
                    # the next state is sampled from a bernoulli distribution 
                    # where the probability is p(x=1) = self._distraction_tp_p1
                    updated_distraction = bernoulli.rvs(size=1, p = self._distraction_tp_p1)[0]
                    self.correction_work = True if updated_distraction == 0 else False
    

            elif action == DiscreteAction.Suggested_Shift_L4:
                self.action_flag = 2
                # Driver choice
                # Categorical distribution, p(j=0,1,2)=[0.05,0.85,0.1], 0-no response, 1-accept, 2-reject
                cate_distribution  = multinomial.rvs(1,np.array(self.response_prob),size=1)
                self.driver_choice = np.where(cate_distribution[0]==1)[0][0]


            elif action == DiscreteAction.Shift_L4:
                self.action_flag = 3
                L4_available = True if obs_now.automation_state[0].LevelMaxNow ==3 else False
                if L4_available:
                    self.turn_L4 = True


            elif action == DiscreteAction.Emergency_CA:
                self.action_flag = 4
                if obs_now.context_state[0].Scenario_Critical == 1:
                    self.risk_clear = True


            elif action == DiscreteAction.Emergency_Stop:
                self.action_flag  = 5
                self.vehicle_stop = True

            if obs_now.driver_state[0].Distraction == 1:
                self.episode_time += 1
            


            ''' state transition conditioned on action'''
            # >> distraction, TTDF, TTDU, and TESD
            if self.correction_work:
                distract_value = 0
                TTDF_value   = 0
                TTDU_value   = 3600
                TESD_value   = 0

            else:
                distract_value = int(self.human_obs[index[0][0],2])
                TTDF_value   = self.tt_obs[index[0][0],0]
                TTDU_value   = self.tt_obs[index[0][0],1]
                TESD_value   = self.tt_obs[index[0][0],5]    


            # >> automation mode      
            if self.turn_L4:
                auto_level = 3
            else:
                auto_level = int(self.auto_obs[index[0][0],0]) 

            
            # >> risk level
            if self.risk_clear:
                sc_value = 0
                TESS_value = 0.0
            else:
                sc_value   = int(self.context_obs[index[0][0],0])
                TESS_value = self.tt_obs[index[0][0],6]

            # >> vehicle status under emergency stop
            vehicle_status = 1 if self.vehicle_stop else 0   
                

            '''states determined by the environment '''
            # driver state
            provider_driver = [] 
            provider_driver.append(                
                DriverState(                
                    Distraction = distract_value, 
                    Fatigue     = int(self.human_obs[index[0][0],3]),
                    Comfort     = int(self.human_obs[index[0][0],4]),
                    Personal_Preference = int(self.human_obs[index[0][0],5]),
                )
            )
                
            # automation state
            provider_automation = []
            provider_automation.append(      
                AutomationState(                
                    Automation_Mode       =  auto_level,
                    LevelMaxNow           =  int(self.auto_obs[index[0][0],1]),
                    LevelMaxNext          =  int(self.auto_obs[index[0][0],2]),      
                )
            )  

            # time metrics state
            provider_tt = []
            provider_tt.append(      
                TimeMetricsState(                
                    TTDF = TTDF_value,
                    TTDU = TTDU_value,
                    TTAF = self.tt_obs[index[0][0],2],
                    TTAU = self.tt_obs[index[0][0],3],
                    TTDD = self.tt_obs[index[0][0],4], 
                    TESD = TESD_value, 
                    TESS = TESS_value,
                )
            )              

            # context state
            provider_context = []
            provider_context.append(
                ContextState(
                    Scenario_Critical = sc_value,
                    Uncomfort_Event   = int(self.context_obs[index[0][0],1]),
                    Vehicle_Stop      = vehicle_status,
                    Time_Stamp        = self.episode_time,
                ))


            # Feedback state
            provider_fb = []
            provider_fb.append(
                FeedBackState(
                    Driver_Choice   = self.driver_choice,
                    Action_Status   = self.action_flag,
                )
            )


        # All states
            provider_state = []
            provider_state = ProviderState(         
                driver_state     = provider_driver,
                automation_state = provider_automation,    
                tt_state         = provider_tt,
                context_state    = provider_context,
                fb_state         = provider_fb,
            )

        
        return provider_state, self.epsicode_ending
    
    


    
    def teardown(self):
    
        self._cum_time = 0.0
    
    
    
    
    




















            
## Example to use
#        
#_Scenario            = ScenarioSequence(scenario_type,step_size)
#segdata              =_Scenario.generate_sequence()
#
#dt = 0.2
#_Scenario_dynamic = ScenarioDynamic(segdata,dt)
#
#for i in range(200):
#    
#    distraction_signal =_Scenario_dynamic.scenario_step()
#    
#    print('distraction_signal = ',distraction_signal)


            ##-----------------------------plot the sequence--------------------------##

            # plt.figure(1)

            # plt.subplot(221)
            # plt.plot(time_seg,np.array(_distraction_segment)[:,2])
            # plt.ylabel('Distracted')

            # plt.subplot(222)
            # plt.plot(time_seg,_TTDF)
            # plt.ylabel('TTDF (s)')
            # plt.xlabel('Time (s)')

            # plt.subplot(223)
            # plt.plot(time_seg,_TTDD)
            # plt.ylabel('TTDD (s)')
            # plt.xlabel('Time (s)')

            # plt.subplot(224)
            # plt.plot(time_seg,_TTDU)
            # plt.ylabel('TTDU (s)')
            # plt.xlabel('Time (s)')

            # plt.tight_layout()
            # plt.show()