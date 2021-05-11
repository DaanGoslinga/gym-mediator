# This is a manual DQN framework running in mediator simulator
# We also provide OpenAI baselines: deepq, for mediator simulator
import os
from typing import Dict, List, Tuple
import gym_mediator
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import *
import random
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs



class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32) # one action for each sample
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0


    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self) -> Dict[str, np.ndarray]:

        idxs = np.random.choice(self.size, size=self.batch_size, replace=False) # sample 1*batch_size

        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size



class Network(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()
        self.hidden_size = 128
        self.layers = nn.Sequential(
            nn.Linear(in_dim, self.hidden_size), 
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size), 
            nn.ReLU(), 
            nn.Linear(self.hidden_size, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)





class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value aof epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
    """

    def __init__(
        self, 
        env: gym.Env,
        replay_size: int,
        batch_size: int,
        target_update: int,
        update_after: int,
        update_every: int,
        logger_kwargs,
    ):


        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)

        # obs_dim = len(env.observation_space.spaces)
        # action_dim = env.action_space.n
        obs_dim = 5
        action_dim = 3
        

        self.env = env
        self.replaybuffer = ReplayBuffer(obs_dim, replay_size, batch_size)
        self.batch_size = batch_size
        
        self.epsilon = 0.1
        
    

        self.target_update = target_update
        self.gamma = 0.9

        self.update_after  = update_after
        self.update_every  = update_every
        
        

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
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()     

        
        # optimizer, only for self.dqn
        self.optimizer = Adam(self.dqn.parameters(), lr = 0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=5,gamma = 0.8)
        
        # Set up model saving
        self.logger.setup_pytorch_saver(self.dqn)
        
        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

        
    
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
            

    def get_action(self, state: np.ndarray, test_stage) -> np.ndarray:
        """ Select an action from the input state based on the rl policy: 
        epsilon greedy policy
        """
        
        if not test_stage and (np.random.random() < self.epsilon):
            # selected_action = self.env.action_space.sample()
            action_type, selected_action = random.choice(list(self.sub_action_dict.items()))
            
        else:

            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()

            # selected_action = selected_action.detach().cpu().numpy() 
            # TODO +2
            selected_action = selected_action.detach().cpu().numpy() + 2

        return selected_action







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
        



    def run(self, num_epoch: int, episodes_per_epoch: int, test_episodes: int):
        """Train the agent.
        episode_reward: lists of episode rewards
        sum_rew: float: reward of each iteration
        iter_time: the number of time steps
        update_cnt: determine the frequency of updating the target network

        """
        print('********************* Training Starts *********************')

        self.is_test = False

        model_loss, epsilons = [], []
        episode_id, step_id, iter_time, update_cnt  = 0, 0, 0, 0
        sum_rew = 0.0
        
        state = self.env.reset()
        last_state = state
        already_starts = False
        episode_begin = 1e8
        total_episode = num_epoch * episodes_per_epoch

        while episode_id<total_episode:  
        
            step_id+= 1 
            cus_state = self.custom_obs(state)

            # 1: >>>> Choose action
            action = self.simple_case_action(state)
            if action == -10: 
                if iter_time<2000:
                    action_type, action = random.choice(list(self.sub_action_dict.items()))
                else:
                    action = self.get_action(cus_state, False)
            
            # 2: >>>> Run episode 
            next_state, reward, done, d_info = self.env.step(action)

            # 3: >>>> To determine the moment when data starts to be saved in the replay buffer
            mydict = self.action_dict
            d_att_last = int(last_state[0][0])
            d_att_now = int(state[0][0])
            d_att_next = int(next_state[0][0])
            l4_next = 1 if int(next_state[0][5])==3 else 0

            degradation_begin =  d_att_now==0 and d_att_next==1
            L4_available  = int(state[0][5])==3

            TESD = state[0][12]

            if degradation_begin and not already_starts:
                episode_begin = step_id 
                already_starts = True    

            if not self.is_test and step_id>=episode_begin:
                if d_att_now==1: 
                    # put distraction states into replay buffer
                    cus_next_state = self.custom_obs(next_state)
                    self.transition = [cus_state, action, reward, cus_next_state, done]
                    self.replaybuffer.store(*self.transition)  
                    iter_time = iter_time+1
                    print('Episode:{}, Step:{}, Iteration:{}, State[d_att,cd_activate,L4_available,ssl4_activate,f_dc]:{}'.format(episode_id, step_id, iter_time, cus_state[0]))
                    print('Dis_Last:{}, Dis_Now:{}, Dis_Next:{},L4_Next:{}, Reward+Cost:{}, Action:{}'.format(d_att_last, d_att_now, d_att_next, l4_next, reward, list(mydict.keys())[list(mydict.values()).index(action)]))
                    
    
            # 4: >>>> Update state and sum of rewards

            state      = next_state
            sum_rew += reward
            
            # 5. >>>> End and Reset
            if done:
                print('Done infos: ',d_info)
                print('Return(Sum of Rewards):{}'.format(round(sum_rew,1)))
                print('-------------------------------------------------------------------------------------------------------------------------')

                # TODO
                self.logger.store(EpRet=sum_rew)
               
                # reset env
                state = self.env.reset()
                last_state = state
                sum_rew = 0.0
                episode_id = episode_id+1
                step_id = 0
                already_starts = False
                episode_begin = 1e8

                         
            # 6. >> Update Model Parameters
            if  (iter_time >= self.update_after) and (iter_time % self.update_every ==0): 
                for j in range(self.update_every):
                    self.update_model()
                    update_cnt += 1
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update() 


            # 7. Save and log information
            if (iter_time >= self.update_after and done) and (episode_id+1) % episodes_per_epoch == 0:

                # Epoch information
                epoch = episode_id // episodes_per_epoch

                self.scheduler.step()
                # self.lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

                # Save model
                self.logger.save_state({'env': self.env}, None)

                # Test the performance of the agent
                self.test_agent(test_episodes)

                # Save important info
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('TestEpRet', with_min_and_max=True)
                self.logger.log_tabular('QVals', with_min_and_max=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('TotalEnvInteracts', iter_time)
                self.logger.dump_tabular()

    def update_model(self):
        """Update the model by gradient descent."""

        samples = self.replaybuffer.sample_batch()
        loss_q, q_info = self._compute_dqn_loss(samples)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        self.logger.store(LossQ=loss_q.item(), **q_info)



    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())



    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:

        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise 
        curr_q_value = self.dqn(state).gather(1, action)

        next_q_value = self.dqn_target(next_state).gather(  # Double DQN
            1, self.dqn(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)  # ground truth

        # calculate dqn loss
        loss_fun = torch.nn.MSELoss().to(self.device)
        loss_q = loss_fun(curr_q_value, target)
        loss_info = dict(QVals=curr_q_value.detach().numpy())

        return loss_q, loss_info       
          
    def test_agent(self,test_episodes):

        """ Test the agent """
        for j in range(test_episodes):
            sum_rew, done, state = 0.0, False, self.env.reset()
            while not done:
                action    = self.simple_case_action(state)
                if action == -10:
                    cus_state = self.custom_obs(state)
                    action = self.get_action(cus_state, True)
                next_state, reward, done, infos = self.env.step(action)
                state = next_state
                sum_rew += reward

            self.logger.store(TestEpRet=sum_rew)
        
        
        
   

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='mediator-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)

    parser.add_argument('--exp_name', type=str, default='experiment-dqn-00')

    parser.add_argument('--replay_size', type=int, default=int(500))  # important

    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--episodes_per_epoch', type=int, default=500)

    parser.add_argument('--update_after', type=int, default= 600,
        help= 'update model since then') 

    parser.add_argument('--update_every', type=int, default=50,
        help = 'update frequency for Q network') 

    parser.add_argument('--target_update', type=int, default=100, 
        help='update frequency for target Q network')  

    parser.add_argument('--test_episodes', type=float, default=50, 
        help='the number of test episodes')  

    
    args = parser.parse_args()
    
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    agent = DQNAgent(gym.make(args.env), args.replay_size, args.batch_size, args.target_update,
        args.update_after, args.update_every, logger_kwargs)

    agent.run(args.epochs, args.episodes_per_epoch, args.test_episodes)
