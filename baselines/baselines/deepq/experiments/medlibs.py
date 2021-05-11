'''
To Customize Observations
'''
from gym import spaces
import numpy as np

class MedLibs:
    def __init__(self):

        self.action_dict = {
            'Do_Nothing' : 0,
            'Emergency_CA': 1,
            'Suggested_Shift_L4': 2,
            'Shift_L4': 3,
            'Correct_Distraction': 4,
        }

        self.sub_action_dict = {
            'Suggested_Shift_L4': 2,
            'Shift_L4': 3,
            'Correct_Distraction': 4,
        }

        # TODO: Define a subset of observation space instead
        # observation: d_att,cd_activate,L4_available,ssl4_activate,f_dc
        obs_low  = np.array([0,0,0,0,-1],dtype = np.float32)
        obs_high = np.array([1,1,1,1,2],dtype = np.float32)

        self.subobs_space = spaces.Box(obs_low, obs_high, dtype = np.float32)

        self.sub_act_dim = 3
        



    def custom_obs(self, state): 
        '''
        Return customized the state
        '''
        d_att = int(state[0][0])
        f_dc = int(state[0][18])   # 0 - no reponse; 1 - accept;  2 - reject; -1-inactivated
        f_as = int(state[0][19]) 
        auto_mode  = int(state[0][4])
        L_max_now  = int(state[0][5])  

        cd_activate = 1 if f_as==1 else 0
        L4_available = 1 if L_max_now==3 else 0
        ssl4_activate = 1 if f_as==2 else 0

        c_state = []
        c_state.append([d_att,cd_activate,L4_available,ssl4_activate,f_dc])

        return np.array(c_state)


    def simple_case_action(self, state: np.ndarray)-> np.ndarray:

        '''
        Choose action in normal and critical situation
        '''
        # observations
        d_att = int(state[0][0])
        d_pp  = int(state[0][3])
        auto_mode  = int(state[0][4])
        c_sc  = int(state[0][14])
        

        driver_fit_manual =  auto_mode==0 and d_att==0
        collision_risk    =  c_sc==1
        with_preference   =  d_pp==1

        # Initialization of action
        # if action ==-10, then it is not normal nor critical. 
        action = -10 

        # Status 1: Critical
        if collision_risk:
            action = self.action_dict['Emergency_CA']          # DiscreteAction.Emergency_CA  
                    
        # Status 2: Normal 
        if (driver_fit_manual and not with_preference) and (not collision_risk):
            action = self.action_dict['Do_Nothing']            # DiscreteAction.Do_Nothing   

        # Status: SSL4 feedback
        f_dc = int(state[0][18])   # 0 - no reponse; 1 - accept;  2 - reject
        f_as = int(state[0][19])   # 0 - do nothing; 1 - correct; 2 - suggest shift to L4

        # SSL4 and feedbacks
        SSL4_last_time   = f_as==2
        no_response_ssl4 = f_dc==0 and SSL4_last_time
        accept_ssl4      = f_dc==1 and SSL4_last_time
        reject_ssl4      = f_dc==2 and SSL4_last_time

        if no_response_ssl4:
            # <No Response> and then repreat SSL4
            action = self.action_dict['Suggested_Shift_L4']      # Suggested Shift L4

        elif accept_ssl4:
            # <Accept>  and Shift to L4
            action = self.action_dict['Shift_L4']                # Shift_L4
        
        elif reject_ssl4:
            # <Reject> and Repeat Correction Distraction
            action = self.action_dict['Correct_Distraction']     # Correct Distraction


        return action
    
    def complex_case_action(self, last_state: np.ndarray, state: np.ndarray) -> np.ndarray:

        '''
        Choose action using using decision trees in complex situation
        '''
        # current observations
        d_att = int(state[0][0])
        d_fat = int(state[0][1])
        comfort = int(state[0][2])
        d_pp  = int(state[0][3])

        auto_mode  = int(state[0][4])
        L_max_now  = int(state[0][5])
        L_max_next = int(state[0][6])

        c_sc = int(state[0][14])
        c_ue = int(state[0][15])
        c_vs = int(state[0][16])

        f_dc = int(state[0][18])   # 0 - no reponse; 1 - accept;  2 - reject
        f_as = int(state[0][19])   # 0 - do nothing; 1 - correct; 2 - suggest shift to L4

        # last observations
        last_d_att = int(last_state[0][0])

        # initial values
        action = -10 

        #--------------------------------------------------------#
        # Status 3: Degraded driver behavior

        # Correction
        distraction_begin = last_d_att == 0 and d_att == 1
        DN_last_time = f_as==0

        if distraction_begin and DN_last_time: 
            action = self.action_dict['Correct_Distraction']          

        CD_last_time = f_as==1
        distraction_eliminate = last_d_att == 1 and d_att == 0
        distraction_still     = last_d_att == 1 and d_att == 1
        L4_available = L_max_now == 3 

        if CD_last_time and distraction_eliminate:
            # Correction works
            action = self.action_dict['Do_Nothing']                     

        elif CD_last_time and distraction_still:
            # Correction fails
            if L4_available and f_dc !=2:
                # L4_available and not being rejected before
                action = self.action_dict['Suggested_Shift_L4']         

            elif not L4_available:
                # L4 unavailable and thus repeat corrections
                action = self.action_dict['Correct_Distraction']   

        # SSL4 and feedbacks
        SSL4_last_time   = f_as==2
        no_response_ssl4 = f_dc==0 and SSL4_last_time
        accept_ssl4      = f_dc==1 and SSL4_last_time
        reject_ssl4      = f_dc==2 and SSL4_last_time

        if no_response_ssl4:
            # <No Response> and then repreat SSL4
            action = self.action_dict['Suggested_Shift_L4']      # Suggested Shift L4

        elif accept_ssl4:
            # <Accept>  and Shift to L4
            action = self.action_dict['Shift_L4']                # Shift_L4

        elif reject_ssl4:
            # <Reject> and Repeat Correction Distraction
            action = self.action_dict['Correct_Distraction']     # Correct Distraction

        #-------------------------------------------------------#
        if action == -10:
            print('Warning: no action generated from rule-based algorithm!')
            action_type, action = random.choice(list(self.sub_action_dict.items()))

        return action
    

    def action_env_Q(self,action):
        # from env_action to Q action
        return action - 2

    def action_Q_env(self,action):
        # from Q action to env_action
        return action + 2





