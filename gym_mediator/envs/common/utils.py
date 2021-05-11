from enum import Enum
from typing import List
from dataclasses import dataclass, field
import pyglet



####################### <Action> # #######################
class DiscreteAction(Enum):

    Do_Nothing               = 0
    Emergency_CA             = 1
    Suggested_Shift_L4       = 2
    Shift_L4                 = 3
    Correct_Distraction      = 4


####################  <Observations> # ##########################

@dataclass
class DriverState:
    '''
    Define driver state
    '''
    Distraction: int    
    Fatigue: int 
    Comfort: int
    Personal_Preference: int


@dataclass
class AutomationState:
    '''
    Define automation state
    '''
    
    Automation_Mode: int
    LevelMaxNow: int 
    LevelMaxNext: int



@dataclass
class TimeMetricsState:
    '''
    TT(DF,DU,AF,AU,DD)
    TESD: time elapsed since degraded bebehavior starts

    ''' 
    TTDF: float 
    TTDU: float 
    TTAF: float 
    TTAU: float 
    TTDD: float 
    TESD: float
    TESS: float



@dataclass
class ContextState:
    '''
    Define context state
    '''

    Scenario_Critical: int
    Uncomfort_Event: int
    Vehicle_Stop: int
    Time_Stamp: int

 
    
@dataclass
class FeedBackState:
    '''
    Define driver_response, action_remain, action_suggest, action_correct
    '''
    Driver_Choice: int
    Action_Status: int



@dataclass
class ProviderState:
    '''
    define all environmental observations 
    '''
    
    driver_state: List[DriverState]         = field(default_factory=list)
    automation_state: List[AutomationState] = field(default_factory=list)
    tt_state: List[TimeMetricsState]        = field(default_factory=list)
    context_state: List[ContextState]       = field(default_factory=list)
    fb_state: List[FeedBackState]           = field(default_factory=list)
    


######################  <scenario definitions> # ########################


class ScenarioCategory(Enum):

    driver_unfit_distraction       = 0
    driver_unfit_fatigue           = 1
    auto_unfit         = 2
    driver_initiate    = 3
    mediator_initiate  = 4
    prevent_underload  = 5

    
        



#############################################
#  <Time to driver discomfort: > # 
'''
The time to driver discomfort can be derived from context information and the table developed in 
the offline comfort detection approach. While each situation has a different probability of being 
uncomfortable, they all are potentially uncomfortable and should thus evoke a suggestion from the
Mediator system for taking over control. 
This means that the [time to driver discomfort]can simply be calculated 
as the time until such an event is expected to take place.
The uncomfortable events include predictable and unpredictable ones. 
Traffic jam, poor visibility, and continuous driving, can be predicted, starting from -inf to inf,
Incoming call, talk_passenger, drinking, cannot be predicted, thus TTDD starts from 0 to -inf,
'''

class UncomfortableEvent(Enum):

    traffic_jam        = 0
    poor_visibility    = 1
    continuous_driving = 2
    distraction_event  = 3

            

from random import randrange

Uncomfortable_Event_Dict = {

    'traffic_jam': randrange(2,10)*60,
    'poor_visibility': randrange(2,10)*60,
    'continuous_driving':randrange(5,120)*60,
    'distraction_event': 0,

}


class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()
   