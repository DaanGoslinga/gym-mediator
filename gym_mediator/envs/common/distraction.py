
import numpy as np
import random
import os
from scipy import interpolate
import matplotlib.pyplot as plt


class DistractionSequence:
    
    def __init__(self,stepSize):
        
        # self.file_path   = '/Users/localadmin/mediator_dl_agent/gym_mediator/gym_mediator/dataset/Distraction' 
        self.file_path   = './dataset/Distraction'   

        self._sim_step   = stepSize
        # frame rate of the sequence
        self.frame_rate  = 25
        

    def load(self):
        
        file_list = []
        SegDistraction = []
        for i in os.listdir(self.file_path):
            file_list.append(os.path.join(self.file_path,i))
        
        # randomly pickup a distraction file
        index_file          = random.randint(0,len(file_list)-1)
        DistractionSequence = np.loadtxt(file_list[index_file],delimiter=',',skiprows=1) 
        EpisodeID           = DistractionSequence[:,10]
        EpisodeTimeBuffer   = DistractionSequence[:,11]
        EpisodeDistracted   = DistractionSequence[:,12]
         
        EpisodeID_index     = np.hstack(([1],np.diff(EpisodeID)))
        Episode_st          = np.where(EpisodeID_index>0)
        Episode_num         = np.shape(Episode_st)[1]
        
        
        distraction_slot = []
        while len(distraction_slot) == 0:

            # randomly load a sequence
            case_id                = random.randint(0, Episode_num-1)
            Episode_seg            = np.zeros([Episode_num,2],dtype=int) 
            Episode_seg[case_id,0] = Episode_st[0][case_id]        # start of segment
            
            if case_id<Episode_num-1:
                Episode_seg[case_id,1] = Episode_st[0][case_id+1]-1 # end of segment   
            else:
                Episode_seg[case_id,1] = np.shape(EpisodeID)[0]-1

            Episode_Distracted = EpisodeDistracted[Episode_seg[case_id,0]:Episode_seg[case_id,1]+1]
            distraction_slot   = np.where(Episode_Distracted>0)[0]   

            
        Episode_length     = Episode_seg[case_id,1]- Episode_seg[case_id,0] + 1
        Episode_Time       = np.arange(0,Episode_length,1)/self.frame_rate  # unit: sec
        Episode_TimeBuffer = EpisodeTimeBuffer[Episode_seg[case_id,0]:Episode_seg[case_id,1]+1]

        #interpolation for the step size

        f1 = interpolate.interp1d(Episode_Time, Episode_TimeBuffer, kind='slinear')
        f2 = interpolate.interp1d(Episode_Time, Episode_Distracted, kind='nearest')
        new_time = list(np.arange(0,Episode_Time[-1],self._sim_step))
        new_time_buffer = list(f1(new_time))
        new_distracted  = list(f2(new_time))

        
        # plot the interpolation
        # plt.figure()
        # plt.subplot(211)
        # plt.plot(Episode_Time,Episode_TimeBuffer,'b',new_time,new_time_buffer,'r')
        
        # plt.subplot(212)
        # plt.plot(Episode_Time,Episode_Distracted,'b',new_time,new_distracted,'r')
        # plt.show()

        # outputs

        for i in range(len(new_time)):
            SegDistraction.append([new_time[i],new_time_buffer[i],new_distracted[i]])
        

        
        # #print('DistractionFile: {}, CaseID: {}, Duration: {}s, TimeStep: {}s'.format(index_file,case_id,sim_time,self._sim_step))


        return SegDistraction













