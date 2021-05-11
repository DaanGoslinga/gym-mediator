import os
import tempfile
import gym
import gym_mediator
from gym import spaces
import random
import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common import models
from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func
from medlibs import MedLibs

class ActWrapper(object):
    '''
    act: 
    '''
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    
    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        '''
        To make the instance behave like functions
        e.g., fun(arg1,arg2) is a shorthand for x.__call__(arg1,arg2)
        '''

        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        '''
        To choose action according to observations
        '''
        # DQN doesn't use RNNs so we ignore states and masks

        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)

def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)

def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=5,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    batch_size: int
        size of a batch sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the trained model from. (default: None)(used in test stage)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.

    """

    # Create all the functions necessary to train the model
    sess = get_session()
    set_global_seeds(seed)
    med_libs= MedLibs()

    '''Define Q network 
    inputs: observation place holder(make_obs_ph), num_actions, scope, reuse
    outputs(tensor of shape batch_size*num_actions): values of each action, Q(s,a_{i})
    '''
    q_func = build_q_func(network, **network_kwargs)


    '''  To put observations into a placeholder  '''
    # TODO: Can only deal with Discrete and Box observation spaces for now
    # observation_space = env.observation_space (default)
    # Use sub_obs_space instead

    observation_space = med_libs.subobs_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)


    '''  Customize action  '''
    # TODO: subset of action space. 
    action_dim = med_libs.sub_act_dim

    ''' 
    Returns: deepq.build_train()
        act: (tf.Variable, bool, float) -> tf.Variable
            function to select and action given observation.
            act is computed by [build_act] or [build_act_with_param_noise]
        train: (object, np.array, np.array, object, np.array, np.array) -> np.array
            optimize the error in Bellman's equation.
        update_target: () -> ()
            copy the parameters from optimized Q function to the target Q function. 
        debug: {str: function}
            a bunch of functions to print debug data like q_values.
    '''

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=action_dim,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        double_q= True,
        grad_norm_clipping=10,
        param_noise=param_noise
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': action_dim,
    }

    '''Contruct an act object using ActWrapper'''
    act = ActWrapper(act, act_params)

    ''' Create the replay buffer'''
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    
    '''Create the schedule for exploration starting from 1.'''
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    '''
    Initialize all the uninitialized variables in the global scope and copy them to the target network.
    '''
    U.initialize()
    update_target()
    episode_rewards = [0.0]
    saved_mean_reward = None
    
    obs = env.reset()
    sub_obs = med_libs.custom_obs(obs)  # TODO: customize observations
    pre_obs = obs
    reset = True
    mydict = med_libs.action_dict
    already_starts = False

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td
        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True

        elif load_path is not None:
            # load_path: a trained model/policy 
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        ''' Training loop starts'''
        t = 0
        while t < total_timesteps:
            if callback is not None:
                if callback(locals(), globals()):
                    break
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            

            ''' Choose action: take action and update exploration to the newest value
            '''
            # TODO: Mixed action strategy
            # Normal status, action is easily determined by rules, use [obs]
            action = med_libs.simple_case_action(obs)
            # Distraction status, action is determined by Q, with [sub_obs]
            if action ==-10:
                action = act(np.array(sub_obs)[None], update_eps=update_eps, **kwargs)[0]
                action = med_libs.action_Q_env(action)  # TODO:action_Q_env, from Q_action(0~2) to env_action(2~4)
                
            reset = False

            ''' Step action '''
            new_obs, rew, done, d_info = env.step(action)
            d_att_last = int(pre_obs[0][0])
            d_att_now = int(obs[0][0])
            d_att_next = int(new_obs[0][0])
            #TODO: you can customize reward here.

            ''' Store transition in the replay buffer.'''
            pre_obs = obs
            obs     = new_obs
            sub_new_obs = med_libs.custom_obs(new_obs)

            if (d_att_last==0 and d_att_now==1) and not already_starts:
                already_starts = True

            if  already_starts and d_att_now==1: 
                replay_buffer.add(sub_obs, action, rew, sub_new_obs, float(done))   
                episode_rewards[-1] += rew  # Sum of rewards
                t = t + 1
                print('>> Iteration:{}, State[d_att,cd_activate,L4_available,ssl4_activate,f_dc]:{}'.format(t, sub_obs))
                print('Dis_Last:{}, Dis_Now:{}, Dis_Next:{},Reward+Cost:{}, Action:{}'.format(d_att_last, d_att_now, d_att_next, rew, list(mydict.keys())[list(mydict.values()).index(action)]))

            # update sub_obs
            sub_obs = sub_new_obs  
            
            # Done and Reset
            if done:
                print('Done infos: ',d_info)
                print('======= end =======')
                obs = env.reset()
                sub_obs = med_libs.custom_obs(obs)  # TODO: custom obs
                pre_obs = obs                      # TODO: save obs at t-1
                already_starts = False
                episode_rewards.append(0.0)
                reset = True

            # Update the Q network parameters
            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                # Calculate td-errors
                actions = med_libs.action_env_Q(actions)  # TODO:action_env_Q, from env_action(2~4) to Q_action(0~2)
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically, copy weights of Q to target Q
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act



def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved





def main():

    env = gym.make("mediator-v0")

    act = learn(
        env,
        network=models.mlp(num_layers=3, num_hidden=128, activation=tf.tanh, layer_norm=False),
        lr=1e-3,
        total_timesteps=10000,
        buffer_size=5000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=25,
        batch_size = 128,
        print_freq=100,
        learning_starts=1000,
        gamma = 0.1,
        target_network_update_freq=100,
        param_noise= True,
        callback=callback
    )

    print("Saving model to mediator_model.pkl")
    act.save("mediator-v0_model_00.pkl")


if __name__ == '__main__':
    main()