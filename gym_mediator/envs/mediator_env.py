import math
import gym
from gym.envs.registration import register
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import random
import time
import copy
from gym_mediator.envs.common.mediator import MEDIATOR
from gym_mediator.envs.common.custom import AgentSpec,observation_custom,reward_custom
from gym_mediator.envs.common.utils import ProviderState
from gym_mediator.envs.common.utils import DrawText

# from pyglet import gl
# import pyglet


class MediatorEnv(gym.Env):


    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    
    
    def __init__(self):

        # TODO: the number of actions is set as two: do nothing or corrective
        self.num_action     = 5 

        self.num_level      = 4

        self.case_id        = 0   #   /utils/ScenarioCategory

        self.dt             = 0.1
        
        self._agent_specs   = AgentSpec(
            observation_adapter = observation_custom,
            reward_adapter = reward_custom,
        )

        self.mediator = MEDIATOR(
            time_step    = self.dt,   
            scenario        = self.case_id,
        )

        self.reward_per_step = 0.0 
    
        self._curr_obs: ProviderState = None    
        self._last_obs: ProviderState = None

        self.action_space       = spaces.Discrete(self.num_action)
        self.observation_space  = spaces.Dict(
            {
                "Distraction": spaces.Discrete(2),
                "Fatigue": spaces.Discrete(2),
                "Comfort": spaces.Discrete(2),
                "Personal_Preference":spaces.Discrete(5),
                "AutomationMode": spaces.Discrete(self.num_level),
                "LevelMaxNow": spaces.Discrete(self.num_level),
                "LevelMaxNext": spaces.Discrete(self.num_level),
                "TTDF": spaces.Box(low = np.float32(0), high = np.float32(300), shape=(1,), dtype= np.float32),
                "TTDU": spaces.Box(low = np.float32(0), high = np.float32(300), shape=(1,), dtype= np.float32),
                "TTAF": spaces.Box(low = np.float32(0), high = np.float32(300), shape=(1,), dtype= np.float32),
                "TTAU": spaces.Box(low = np.float32(0), high = np.float32(300), shape=(1,), dtype= np.float32),
                "TTDD": spaces.Box(low = np.float32(0), high = np.float32(300), shape=(1,), dtype= np.float32),
                "TESD": spaces.Box(low = np.float32(0), high = np.float32(300), shape=(1,), dtype= np.float32),
                "TESS": spaces.Box(low = np.float32(0), high = np.float32(300), shape=(1,), dtype= np.float32),
                "SC": spaces.Discrete(2),
                "UC": spaces.Discrete(5),
                "VS": spaces.Discrete(2),
                "TS": spaces.Discrete(10),
                "Driver_Choice": spaces.Discrete(3),
                "Action_Status": spaces.Discrete(6),

            })

        self.state  = None   
        self.viewer = None



        
    def step(self, action):

        '''
        Step Mediator, customize reward and state
        '''
          
        next_obs, env_reward, done, infos  = self.mediator.step(action) 

        self.reward_per_step  = self._agent_specs.reward_adapter(self._last_obs, action, self._curr_obs, next_obs, self.case_id)

        self.state = self._agent_specs.observation_adapter(next_obs)

        self._last_obs = copy.deepcopy(self._curr_obs)
        self._curr_obs = copy.deepcopy(next_obs)
        
        


        return self.state, self.reward_per_step, done, infos
                     
        

    def reset(self):

        '''
        Reset with a new scenario
        '''

        env_obs  = self.mediator.reset()

        self._last_obs = copy.deepcopy(env_obs)
        self._curr_obs = self._last_obs

        self.state = self._agent_specs.observation_adapter(self._curr_obs) 

        
        return self.state
    

    


            
    # def render(self,mode = 'human'):

        
    #     screen_width, screen_height = 900, 600
    #     VP_W, VP_H = 450, 300
        

    #     if self.viewer is None:

    #         self.viewer = rendering.Viewer(screen_width, screen_height)

    #         self.count_step = 0

    #         ##----------------------- TTDF----------------------##

    #         # Draw Bars
    #         l,r,t,b = -5,5,65,-65
    #         a_x = 150
    #         bar_a = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_a.set_color(.8,.6,.4)
    #         bar_a.add_attr(rendering.Transform(translation=(a_x, 450)))
    #         self.viewer.add_geom(bar_a)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,5,-5
    #         marker_a = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+5,t-5), (r,b)])
    #         marker_a.set_color(95/255,158/255,160/255)
    #         self.marker_a_trans = rendering.Transform()
    #         marker_a.add_attr(self.marker_a_trans)
    #         self.viewer.add_geom(marker_a)

    #         # ttdf
    #         label_ttdf = pyglet.text.Label('TTDF', font_size=14,
    #                     x=a_x -30, y=530, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_ttdf.draw()
    #         self.viewer.add_geom(DrawText(label_ttdf))

    #         # label -1
    #         label_1 = pyglet.text.Label('10+', font_size=13,
    #                     x=a_x -45, y=498, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('0', font_size=13,
    #                     x=a_x-40, y=380, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))




    #         ##----------------------- TTDU---------------------##
    #         # Draw Bars
    #         l,r,t,b = -5,5,65,-65
    #         b_x = 300
    #         bar_b = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_b.set_color(.8,.6,.4)
    #         bar_b.add_attr(rendering.Transform(translation=(b_x, 450)))
    #         self.viewer.add_geom(bar_b)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,5,-5
    #         marker_b = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+5,t-5), (r,b)])
    #         marker_b.set_color(95/255,158/255,160/255)
    #         self.marker_b_trans = rendering.Transform()
    #         marker_b.add_attr(self.marker_b_trans)
    #         self.viewer.add_geom(marker_b)

    #         # ttdu
    #         label_ttdu = pyglet.text.Label('TTDU', font_size=14,
    #                     x=b_x-30, y=530, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_ttdu.draw()
    #         self.viewer.add_geom(DrawText(label_ttdu))

    #         # label -1
    #         label_1 = pyglet.text.Label('10+', font_size=13,
    #                     x=b_x-48, y=498, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('0', font_size=13,
    #                     x=b_x-43, y=380, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))



    #         ##-----------------------TTDD---------------------##

    #         # Draw Bars
    #         l,r,t,b = -5,5,65,-65
    #         c_x = 450
    #         bar_c = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_c.set_color(.8,.6,.4)
    #         bar_c.add_attr(rendering.Transform(translation=(c_x, 450)))
    #         self.viewer.add_geom(bar_c)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,5,-5
    #         marker_c = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+5,t-5), (r,b)])
    #         marker_c.set_color(95/255,158/255,160/255)
    #         self.marker_c_trans = rendering.Transform()
    #         marker_c.add_attr(self.marker_c_trans)
    #         self.viewer.add_geom(marker_c)


    #         # ttdd
    #         label_ttdd = pyglet.text.Label('TTDD', font_size=14,
    #                     x=c_x-30, y=530, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_ttdd.draw()
    #         self.viewer.add_geom(DrawText(label_ttdd))

    #         # label -1
    #         label_1 = pyglet.text.Label('10+', font_size=13,
    #                     x=c_x-45, y=498, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('-10', font_size=13,
    #                     x=c_x-45, y=380, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))



    #         ##-----------------------TTAF---------------------##

    #         # Draw Bars
    #         l,r,t,b = -5,5,65,-65
    #         d_x = 450
    #         d_y = 240
    #         bar_d = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_d.set_color(.8,.6,.4)
    #         bar_d.add_attr(rendering.Transform(translation=(d_x, d_y)))
    #         self.viewer.add_geom(bar_d)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,5,-5
    #         marker_d = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+5,t-5), (r,b)])
    #         marker_d.set_color(95/255,158/255,160/255)
    #         self.marker_d_trans = rendering.Transform()
    #         marker_d.add_attr(self.marker_d_trans)
    #         self.viewer.add_geom(marker_d)

    #         # ttaf
    #         label_ttaf = pyglet.text.Label('TTAF', font_size=14,
    #                     x=d_x-30, y=d_y+80, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_ttaf.draw()
    #         self.viewer.add_geom(DrawText(label_ttaf))

    #         # label -1
    #         label_1 = pyglet.text.Label('10+', font_size=13,
    #                     x=d_x -45, y=d_y+48, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('0', font_size=13,
    #                     x=d_x-40, y=d_y-70, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))



    #         ##-----------------------TTAU---------------------##

    #         # Draw Bars
    #         l,r,t,b = -5,5,65,-65
    #         e_x = 600
    #         e_y = 240
    #         bar_e = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_e.set_color(.8,.6,.4)
    #         bar_e.add_attr(rendering.Transform(translation=(e_x, e_y)))
    #         self.viewer.add_geom(bar_e)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,5,-5
    #         marker_e = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+5,t-5), (r,b)])
    #         marker_e.set_color(95/255,158/255,160/255)
    #         self.marker_e_trans = rendering.Transform()
    #         marker_e.add_attr(self.marker_e_trans)
    #         self.viewer.add_geom(marker_e)

    #         # ttaf
    #         label_ttau = pyglet.text.Label('TTAU', font_size=14,
    #                     x=e_x-30, y=e_y+80, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_ttau.draw()
    #         self.viewer.add_geom(DrawText(label_ttau))

    #         # label -1
    #         label_1 = pyglet.text.Label('10+', font_size=13,
    #                     x=e_x -45, y=e_y+48, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('0', font_size=13,
    #                     x=e_x-40, y=e_y-70, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))


    #         ##-----------------Automation Mode---------------------##

    #         # Draw Bars
    #         l,r,t,b = -7,7,65,-65
    #         f_x = 600
    #         bar_f = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_f.set_color(139/255,69/255,19/255)
    #         bar_f.add_attr(rendering.Transform(translation=(f_x, 450)))
    #         self.viewer.add_geom(bar_f)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,7,-7
    #         marker_f = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+7,t-7), (r,b)])
    #         marker_f.set_color(0/255,100/255,0/255)
    #         self.marker_f_trans = rendering.Transform()
    #         marker_f.add_attr(self.marker_f_trans)
    #         self.viewer.add_geom(marker_f)


    #         # ttdd
    #         label_ttdd = pyglet.text.Label('Auto Level', font_size=14,
    #                     x=f_x-40, y=530, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_ttdd.draw()
    #         self.viewer.add_geom(DrawText(label_ttdd))

    #         # label -L4
    #         label_1 = pyglet.text.Label('L4', font_size=13,
    #                     x=f_x-45, y=498, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))


    #         # label - L3
    #         label_0 = pyglet.text.Label('L3', font_size=13,
    #                     x=f_x-45, y=498-39.3, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))


    #         # label - L2
    #         label_0 = pyglet.text.Label('L2', font_size=13,
    #                     x=f_x-45, y=380+39.3, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))

    #         # label - 0
    #         label_0 = pyglet.text.Label('M', font_size=13,
    #                     x=f_x-45, y=380, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))



    #         ##-----------------------Distracted---------------------##

    #         # Draw Bars
    #         l,r,t,b = -5,5,65,-65
    #         r_x = 150
    #         r_y = 240
    #         bar_r = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_r.set_color(.8,.6,.4)
    #         bar_r.add_attr(rendering.Transform(translation=(r_x, r_y)))
    #         self.viewer.add_geom(bar_r)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,5,-5
    #         marker_r = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+5,t-5), (r,b)])
    #         marker_r.set_color(95/255,158/255,160/255)
    #         self.marker_r_trans = rendering.Transform()
    #         marker_r.add_attr(self.marker_r_trans)
    #         self.viewer.add_geom(marker_r)

    #         # distracted
    #         label_dis = pyglet.text.Label('Distracted', font_size=14,
    #                     x=r_x-35, y=r_y+80, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_dis.draw()
    #         self.viewer.add_geom(DrawText(label_dis))

    #         # label -1
    #         label_1 = pyglet.text.Label('T', font_size=13,
    #                     x=r_x -45, y=r_y+48, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('F', font_size=13,
    #                     x=r_x-40, y=r_y-70, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))

    #         ##-----------------------Fatigue---------------------##

    #         # Draw Bars
    #         l,r,t,b = -5,5,65,-65
    #         s_x = 300
    #         s_y = 240
    #         bar_s = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_s.set_color(.8,.6,.4)
    #         bar_s.add_attr(rendering.Transform(translation=(s_x, s_y)))
    #         self.viewer.add_geom(bar_s)

    #         # Draw Bar-Marker

    #         l,r,t,b = -10,10,5,-5
    #         marker_s = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r+5,t-5), (r,b)])
    #         marker_s.set_color(95/255,158/255,160/255)
    #         self.marker_s_trans = rendering.Transform()
    #         marker_s.add_attr(self.marker_s_trans)
    #         self.viewer.add_geom(marker_s)

    #         # distracted
    #         label_fatigue = pyglet.text.Label('Fatigue', font_size=14,
    #                     x=s_x-30, y=s_y+80, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_fatigue.draw()
    #         self.viewer.add_geom(DrawText(label_fatigue))

    #         # label -1
    #         label_1 = pyglet.text.Label('T', font_size=13,
    #                     x=s_x -45, y=s_y+48, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('F', font_size=13,
    #                     x=s_x-40, y=s_y-70, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))

    #         by1,by2 = 182, 298 



    #         # ##-------------------- draw boundary line ------------------------##

    #         # uy1, uy2 = 380, 520
    #         # track_a = rendering.Line((75,uy1), (75,uy2))
    #         # track_a.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_a)

    #         # track_b = rendering.Line((225,uy1), (225,uy2))
    #         # track_b.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_b)

    #         # track_c = rendering.Line((375,uy1), (375,uy2))
    #         # track_c.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_c)

    #         # track_d = rendering.Line((525,uy1), (525,uy2))
    #         # track_d.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_d)


    #         # track_h = rendering.Line((675,uy1), (675,uy2))
    #         # track_h.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_h)


    #         # 
    #         # track_e = rendering.Line((75,by1), (75,by2))
    #         # track_e.set_color(169/255,169/255,169/255)
    #         # track_e.add_attr(LineWidth(10))
    #         # self.viewer.add_geom(track_e)

    #         # track_f = rendering.Line((225,by1), (225,by2))
    #         # track_f.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_f)

    #         # track_g = rendering.Line((375,by1), (375,by2))
    #         # track_g.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_g)


    #         # track_i = rendering.Line((525,by1), (525,by2))
    #         # track_i.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_i)


    #         # track_j = rendering.Line((675,by1), (675,by2))
    #         # track_j.set_color(169/255,169/255,169/255)
    #         # self.viewer.add_geom(track_j)




    #         ##------------------- draw time line-------------------##

    #         # horizontal bottom
    #         l,r,t,b = -240,240,-1,1
    #         t_x = 390
    #         t_y = 25
    #         bar_t = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_t.set_color(0,0,0)
    #         bar_t.add_attr(rendering.Transform(translation=(t_x, t_y)))
    #         self.viewer.add_geom(bar_t)



    #         # horizontal top
    #         l,r,t,b = -240,240,-0.3,0.3
    #         t1_x = 390
    #         t1_y = 84
    #         bar_t1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_t1.set_color(0,0,0)
    #         bar_t1.add_attr(rendering.Transform(translation=(t1_x, t1_y)))
    #         self.viewer.add_geom(bar_t1)



    #         # vertical line
    #         l,r,t,b = -1,1,56,-56
    #         p_x = r_x
    #         p_y = 80
    #         bar_p = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    #         bar_p.set_color(0,0,0)
    #         bar_p.add_attr(rendering.Transform(translation=(p_x, p_y)))
    #         self.viewer.add_geom(bar_p)

    #         r_x = 150
    #         r_y = 230

    #         # label -1
    #         label_1 = pyglet.text.Label('Correct 1', font_size=13,
    #                     x=r_x -80, y=74, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - 0
    #         label_0 = pyglet.text.Label('None 0', font_size=13,
    #                     x=r_x-65, y=18, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_0.draw()
    #         self.viewer.add_geom(DrawText(label_0))


    #         # label - Action
    #         label_1 = pyglet.text.Label('Action', font_size=14,
    #                     x=r_x -70, y=120, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))

    #         # label - Time
    #         label_1 = pyglet.text.Label('Time', font_size=14,
    #                     x=640, y=15, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label_1.draw()
    #         self.viewer.add_geom(DrawText(label_1))


    #         ##------------------------ draw arrow------------------------##
    #         # x, y axis

    #         t_x = 630
    #         t_y = 25
    #         bar_t = rendering.FilledPolygon([(-10,-5), (-5,0), (-10,5),(5,0)])
    #         bar_t.set_color(0,0,0)
    #         bar_t.add_attr(rendering.Transform(translation=(t_x, t_y)))
    #         self.viewer.add_geom(bar_t)


    #         t_x = 150
    #         t_y = 135
    #         bar_t = rendering.FilledPolygon([(-5,-5), (0,10), (5,-5),(0,0)])
    #         bar_t.set_color(0,0,0)
    #         bar_t.add_attr(rendering.Transform(translation=(t_x, t_y)))
    #         self.viewer.add_geom(bar_t)



    #         ## ----------------Action---------------------##
    #         # draw a circle
    #         action_flag = rendering.make_circle(6)
    #         action_flag.set_color(.9,.1,.1)
    #         self.action_trans = rendering.Transform()
    #         action_flag.add_attr(self.action_trans) 
    #         self.viewer.add_geom(action_flag)


    #         ##-----------------Episode and Reward---------##
    #         sc_x, sc_y = 400, 130

    #         #---------#
    #         label = pyglet.text.Label('Reward: ', font_size=14,
    #                     x=sc_x+50, y=sc_y-10, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label.draw()
    #         self.viewer.add_geom(DrawText(label))
            
    #         self.reward_label = pyglet.text.Label('000', font_size=13,
    #         x=sc_x+132, y=sc_y, anchor_x='left', anchor_y='center',
    #         color=(0, 0, 0, 255))

    #         #---------#
    #         label = pyglet.text.Label('Episode: ', font_size=14,
    #                     x=sc_x-80, y=sc_y-10, anchor_x='left', anchor_y='bottom',
    #                     color=(0, 0, 0, 255))
    #         label.draw()
    #         self.viewer.add_geom(DrawText(label))


            
    #         self.episode_label = pyglet.text.Label('000', font_size=13,
    #         x=sc_x, y=sc_y, anchor_x='left', anchor_y='center',
    #         color=(0, 0, 0, 255))






    #     ##----------------------Observations------------------##

    #     print('Hi~')
    #     self.count_step += 1
    #     time.sleep(.2)

    #     TTDF = self.state[0][7]
    #     TTDU = self.state[0][8]
    #     TTAF = self.state[0][9]
    #     TTAU = self.state[0][10]
    #     TTDD = self.state[0][11]

    #     Distract_signal = self.state[0][0]
    #     Fatigue_signal  = self.state[0][1]
    #     Auto_level      = self.state[0][3]

    #     time_clock      = 150 + 1*self.count_step

    #     mediator_action = random.choice(range(2))

    #     ## ------------------Constants---------------------##
    #     a_x = 150
    #     b_x = 300
    #     c_x = 450
    #     d_x = 450
    #     e_x = 600
    #     f_x = 600
    #     r_x = 150
    #     s_x = 300
    #     by1,by2 = 182, 298 


    #     ## ---------------- Translation--------------------##

    #     # move TTDF
    #     if TTDF ==0:
    #         self.marker_a_trans.set_translation(a_x, 392)
    #     else:
    #         self.marker_a_trans.set_translation(a_x, 508)


    #     # move TTDU
    #     if TTDU ==0:
    #         self.marker_b_trans.set_translation(b_x, 392)
    #     else:
    #         self.marker_b_trans.set_translation(b_x, 508)

    #     # move TTDD
    #     if TTDD ==0:
    #         self.marker_c_trans.set_translation(c_x, 392)
    #     else:
    #         self.marker_c_trans.set_translation(c_x, 508)



    #     # move TTAF
    #     if TTAF ==0:
    #         self.marker_d_trans.set_translation(d_x, by1)
    #     else:
    #         self.marker_d_trans.set_translation(d_x, by2)


    #     # move TTAU
    #     if TTAU ==0:
    #         self.marker_e_trans.set_translation(e_x, by1)
    #     else:
    #         self.marker_e_trans.set_translation(e_x, by2)



    #     # move distract
    #     if Distract_signal ==0:
    #         self.marker_r_trans.set_translation(r_x, by1)
    #     else:
    #         self.marker_r_trans.set_translation(r_x, by2)

    #     # move fatigue
    #     if Fatigue_signal ==0:
    #         self.marker_s_trans.set_translation(s_x, by1)
    #     else:
    #         self.marker_s_trans.set_translation(s_x, by2)


    #     # move auto_level
    #     if Auto_level == 0:
    #         self.marker_f_trans.set_translation(f_x, 392)

    #     elif Auto_level == 1:
    #         self.marker_f_trans.set_translation(f_x, 392+39)

    #     elif Auto_level == 2:
    #         self.marker_f_trans.set_translation(f_x, 508-39)

    #     elif Auto_level == 3: 
    #         self.marker_f_trans.set_translation(f_x, 508)


    #     # # action option 1
    #     if mediator_action == 0: # None
    #         self.action_trans.set_translation(time_clock,24)


    #     elif mediator_action == 1:
    #         self.action_trans.set_translation(time_clock,84)

    #     # # action option 2
    #     # if mediator_action == 0: # None

    #     #     action_flag = rendering.make_circle(1)
    #     #     action_flag.set_color(.9,.1,.1)
    #     #     action_flag.add_attr(rendering.Transform(translation=(time_clock,24))) 
    #     #     self.viewer.add_geom(action_flag)

    #     # elif mediator_action == 1:

    #     #     action_flag = rendering.make_circle(1)
    #     #     action_flag.set_color(.9,.1,.1)
    #     #     action_flag.add_attr(rendering.Transform(translation=(time_clock,84))) 
    #     #     self.viewer.add_geom(action_flag)


    #     win = self.viewer.window
    #     win.switch_to()
    #     win.dispatch_events()

    #     win.clear()

    #     self.marker_a_trans.enable()
    #     self.marker_a_trans.disable()
        

    #     self.episode_label.text = "%03i" % self.count_step
    #     self.reward_label.text  = "%03i" % self.reward_per_step

    #     self.viewer.add_geom(DrawText(self.episode_label))
    #     self.viewer.add_geom(DrawText(self.reward_label))
        
    #     win.flip()
    




       
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    #     # input()


        


    def close(self):
        if self.mediator is not None:
            self.mediator.destroy()

        if self.viewer:
            self.viewer.close()
            self.viewer = None







# Register the environment

register(
    id='mediator-v0',
    entry_point='gym_mediator.envs:MediatorEnv',
    max_episode_steps=200,
)





        
        
