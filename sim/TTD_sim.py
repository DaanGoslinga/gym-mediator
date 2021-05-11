# Compute TTDF/TTDU/TTDD
import random
import numpy as np
import matplotlib.pyplot as plt


def compute_ttdx(seq_len, auto_level, distraction_level, fatigue_level, ndrt_type, event_state, dt, np_uc):
    
    '''
    Parameters
    ------------
    + seq_len: int
        Length of the sequence
    + auto_level: ndarray
        Level of automation, i.e., L0/L2/L3/L4
    + distraction_level: ndarray
        Level of distraction, i.e., 0,1
    + fatigue_level: ndarray
        Level of fatigue, i.e., 0,1,2,3
    + ndrt_type: ndarray
        Non-driving related task, which impacts the take over time in L3/l4
    + event_state: ndarray
        Uncomfrotable events
    + dt: float
        Time of each step
    + np_uc: predicted time, second
        Predicted horizon of uncomfortable events
    
    Returns
    -------------
    + TTDF: np.array
        Time to driver fitness, only filled with L3 and L4, otherwise np.nan
    + TTDU: np.array
        Time to driver unfitness, only filled with L0 and L2, otherwise np.nan
    + TTDD: np.array
        Time to driver discomfort, determined by uncomfortable events
    '''
    
    UNCOMFORT_EVENT = {
        0: 'No_Event',
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

    # TTDF
    TTDF   = np.zeros((seq_len,1))
    # distraction related TTDF
    TTDF_d = np.zeros((seq_len,1))  
    # fatigue related TTDF
    TTDF_f = np.zeros((seq_len,1))  
    
    # TTDU
    M = 9999
    TTDU   = M * np.ones((seq_len,1))  
    # distraction related TTDU
    TTDU_d = M * np.ones((seq_len,1))  
    # fatigue related TTDU
    TTDU_f = M * np.ones((seq_len,1))  

    # TTDD
    TTDD = M * np.ones((seq_len,1))    
 
    for i in range(seq_len):
        print('Step: {}'.format(i))
        print('automation mode: ',MODE[auto_level[i]])

        # <TTDU/TTDF>
        if MODE[auto_level[i]] == "L0" or MODE[auto_level[i]] == "L2":
            TTDF[i] = np.nan
            if DISTRACTION[distraction_level[i]]=='Sufficient SA':
                TTDU_d[i] = 60 * 5 
            elif DISTRACTION[distraction_level[i]]=='Long term loss of SA':
                TTDU_d[i] = 0                

            if FATIGUE[fatigue_level[i]]=='Alert':
                TTDU_f[i] = 180 * 60 
            elif FATIGUE[fatigue_level[i]]=='Neither alert nor sleepy':
                TTDU_f[i] = 120 * 60
            elif FATIGUE[fatigue_level[i]]=='Some signs of sleepiness':
                TTDU_f[i] = 30 * 60
            elif FATIGUE[fatigue_level[i]]=='Sleepy':
                TTDU_f[i] = 0

            TTDU[i] = min(TTDU_d[i],TTDU_f[i])

        elif MODE[auto_level[i]] == "L3" or MODE[auto_level[i]] == "L4":
            TTDU[i] = np.nan
            
            TTDF_f[i] = 0 if fatigue_level[i]<3 else M

            if NDRT[ndrt_type[i]]=="None":
                TTDF_d[i] = 0
            elif NDRT[ndrt_type[i]] =="Messaging":
                TTDF_d[i] = 5
            elif NDRT[ndrt_type[i]] =="Obstruction":
                TTDF_d[i] = 10
            elif NDRT[ndrt_type[i]] =="Immersion":
                TTDF_d[i] = 10    
            elif NDRT[ndrt_type[i]] =="Obstruction_Immersion":
                TTDF_d[i] = 20

            TTDF[i] = max(TTDF_d[i],TTDF_f[i])

        # <TTDD>
        uc_end = min(seq_len,i+np_uc)        
        if len(np.where(event_state[i:uc_end]>=1)[0])==0:
            print('Uncomfortable Event Not Found!')
            TTDD[i] = M
        else:
            TTDD[i] = dt * min(np.where(event_state[i:uc_end]>=1)[0])

        # summary
        print('TTDF',TTDF[i])
        print('TTDU',TTDU[i])
        print('TTDD',TTDD[i])

        

    return TTDF, TTDU, TTDD











if __name__ == '__main__':

    seq_len = 80
    dt = 1
    np_uc = 180

    # automation level
    j1 = 0 * np.ones((40))
    j2 = 2 * np.ones((40))
    auto_level = np.hstack((j1,j2))

    # distraction 
    a1 = 0 * np.ones((10))
    a2 = 0 * np.ones((10))
    a3 = 1 * np.ones((10))
    a4 = 0 * np.ones((10))
    a5 = 0 * np.ones((10))
    a6 = 1 * np.ones((10)) 
    a7 = 1 * np.ones((10)) 
    a8 = 0 * np.ones((10)) 
    temp1 = np.hstack((np.hstack((a1,a2)),a3))
    temp2 = np.hstack((temp1,a4))
    temp3 = np.hstack((temp2,a5))
    temp4 = np.hstack((temp3,a6))
    temp5 = np.hstack((temp4,a7))
    distraction_level = np.hstack((temp5,a8))  

    # fatigue
    a1 = 0 * np.ones((10))
    a2 = 0 * np.ones((10))
    a3 = 1 * np.ones((10))
    a4 = 2 * np.ones((10))
    a5 = 3 * np.ones((10))
    a6 = 0 * np.ones((10)) 
    a7 = 0 * np.ones((10)) 
    a8 = 0 * np.ones((10)) 
    temp1 = np.hstack((np.hstack((a1,a2)),a3))
    temp2 = np.hstack((temp1,a4))
    temp3 = np.hstack((temp2,a5))
    temp4 = np.hstack((temp3,a6))
    temp5 = np.hstack((temp4,a7))
    fatigue_level = np.hstack((temp5,a8))  

    # ndrt 
    a1 = 0 * np.ones((10))
    a2 = 0 * np.ones((10))
    a3 = 0 * np.ones((10))
    a4 = 0 * np.ones((10))
    a5 = 1 * np.ones((10))
    a6 = 1 * np.ones((10)) 
    a7 = 0 * np.ones((10)) 
    a8 = 0 * np.ones((10)) 
    temp1 = np.hstack((np.hstack((a1,a2)),a3))
    temp2 = np.hstack((temp1,a4))
    temp3 = np.hstack((temp2,a5))
    temp4 = np.hstack((temp3,a6))
    temp5 = np.hstack((temp4,a7))
    ndrt_type = np.hstack((temp5,a8))  


    # uncomfortable event  
    b1 = 0 * np.ones((10))
    b2 = 0 * np.ones((10))
    b3 = 0 * np.ones((10))
    b4 = 1 * np.ones((10))
    b5 = 1 * np.ones((10))
    b6 = 0 * np.ones((10)) 
    b7 = 0 * np.ones((10)) 
    b8 = 2 * np.ones((10)) 
    tempb1 = np.hstack((np.hstack((b1,b2)),b3))
    tempb2 = np.hstack((tempb1,b4))
    tempb3 = np.hstack((tempb2,b5))
    tempb4 = np.hstack((tempb3,b6))
    tempb5 = np.hstack((tempb4,b7))
    event_state = np.hstack((tempb5,b8))  

    TTDF, TTDU, TTDD = compute_ttdx(seq_len, auto_level, distraction_level, fatigue_level, ndrt_type, event_state, dt, np_uc)


    # TTDU/TTDF
    plt.figure(figsize=(15,8))
    plt.subplot(231)
    plt.plot(auto_level,'k.')
    plt.ylabel('auto level')
    plt.xlabel('time [s]')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)

    plt.subplot(232)
    plt.plot(distraction_level,'b.')
    plt.ylabel('distraction level')
    plt.yticks([0, 1, 2, 3], ['0','1','2','3']) 
    plt.xlabel('time [s]')
    plt.xlim(0,80)
    plt.grid(True)

    plt.subplot(233)
    plt.plot(fatigue_level,'c.')
    plt.ylabel('fatigue level')
    plt.yticks([0, 1, 2, 3], ['0','1','2','3']) 
    plt.xlabel('time [s]')
    plt.xlim(0,80)
    plt.grid(True)


    plt.subplot(234)
    plt.plot(ndrt_type,'m.')
    plt.ylabel('ndrt')
    plt.xlabel('time [s]')
    plt.yticks([0, 1, 2, 3, 4], ['0','1','2','3','4']) 
    plt.xlim(0,80)
    plt.grid(True) 

    plt.subplot(235)
    plt.plot(np.log10(TTDU+0.01*np.ones((seq_len,1))),'r')
    plt.ylabel('log(TTDU)')
    plt.xlabel('time [s]')
    plt.xlim(0,80)
    plt.grid(True)
    
    plt.subplot(236)
    plt.plot(np.log10(TTDF+0.01*np.ones((seq_len,1))),'g')
    plt.ylabel('log(TTDF)')
    plt.xlabel('time [s]')
    plt.xlim(0,80)
    plt.grid(True)
    plt.show()   




    # TTDD
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.plot(auto_level,'k.')
    plt.ylabel('auto level')
    plt.xlabel('time [s]')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)

    plt.subplot(222)
    plt.plot(event_state,'b.')
    plt.ylabel('uncomfortable events')
    plt.yticks([0, 1, 2, 3, 4, 5], ['0','1','2','3','4','5']) 
    plt.xlabel('time [s]')
    plt.xlim(0,80)
    plt.grid(True)

    plt.subplot(212)
    plt.plot(np.log10(0.01*np.ones((seq_len,1))+TTDD),'r.')
    plt.ylabel('log(TTDD)')
    plt.xlabel('time [s]')
    plt.xlim(0,80)
    plt.grid(True)
    plt.show()




            