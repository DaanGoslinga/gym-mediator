
# This code shows how TTAF is computed.


import random
import numpy as np
import matplotlib.pyplot as plt


def compute_ttax(seq_len, auto_level, level_max, dt):
    '''
    Parameters
    -----------
    + seq_len: int
        length of the sequence
    + auto_level: np.array
        level of automation mode
    + level_max:  np.array
        maximum automation level
    + dt: float
        time step

    Return
    --------
    TTA2F, TTA3F, TTA4F: np.array
        Time to automation fitness. Default value is a large number M = 9999

    TTA2U, TTA3U, TTA4U: np.array
        Time to automation unfitness. 
        Default value is a large number M = 9999. 
        It is undefined (i.e., TTA2U/TTA3U/TTA4U=np.nan) when TTA2F/TTA3F/TTA4F is unavailable

    '''
    
    MODE = {
        0: 'L0',
        1: 'L2', 
        2: 'L3',
        3: 'L4',
    }

    # Initialization
    M = 9999  
    TTA2F = M * np.ones((seq_len,1))
    TTA3F = M * np.ones((seq_len,1))
    TTA4F = M * np.ones((seq_len,1))
    TTA2U = M * np.ones((seq_len,1))
    TTA3U = M * np.ones((seq_len,1))
    TTA4U = M * np.ones((seq_len,1))

    # Simulation
    for i in range(seq_len):
        print('Step: {}'.format(i))
        # L0
        if MODE[auto_level[i]] == "L0":
            print('automation mode: L0')
            if np.amax(level_max[i:])==0:
                print('warning: automation unavailable!')
                TTA2U[i] = np.nan
                TTA3U[i] = np.nan
                TTA4U[i] = np.nan
            else:
                if len(np.where(level_max[i:]>=1)[0]):
                    index_l2 = min(np.where(level_max[i:]>=1)[0])
                    TTA2F[i] = dt * index_l2 
                    if len(np.where(level_max[i:]<1)[0]):
                        temp_a2u = np.where(level_max[i:]<1)[0]
                        TTA2U[i] = dt * temp_a2u[min(np.where(temp_a2u>index_l2)[0])]
                else:
                    TTA2U[i] = np.nan

                if len(np.where(level_max[i:]>=2)[0]):
                    index_l3 = min(np.where(level_max[i:]>=2)[0])
                    TTA3F[i] = dt * index_l3
                    if len(np.where(level_max[i:]<2)[0]):
                        temp_a3u = np.where(level_max[i:]<2)[0]
                        TTA3U[i] = dt * temp_a3u[min(np.where(temp_a3u>index_l3)[0])]
                else:
                    TTA3U[i] = np.nan

                if len(np.where(level_max[i:]==3)[0]):
                    index_l4 = min(np.where(level_max[i:]==3)[0])
                    TTA4F[i] = dt * index_l4 
                    if len(np.where(level_max[i:]<3)[0]):
                        temp_a4u = np.where(level_max[i:]<3)[0]
                        TTA4U[i] = dt * temp_a4u[min(np.where(temp_a4u>index_l4)[0])]
                else:
                    TTA4U[i] = np.nan

        # L2
        elif MODE[auto_level[i]] == "L2":
            print('automation mode: L2')
            TTA2F[i]  = 0   
            index_l2u = min(np.where(level_max[i:]<1)[0])
            TTA2U[i]  = dt * index_l2u
 
            if len(np.where(level_max[i:]>=2)[0]):
                index_l3 = min(np.where(level_max[i:]>=2)[0])
                TTA3F[i] = dt * index_l3
                if len(np.where(level_max[i:]<2)[0]):
                    temp_a3u = np.where(level_max[i:]<2)[0]
                    TTA3U[i] = dt * temp_a3u[min(np.where(temp_a3u>index_l3)[0])]
            else:
                TTA3U[i] = np.nan

            if len(np.where(level_max[i:]==3)[0]):
                index_l4 = min(np.where(level_max[i:]==3)[0])
                TTA4F[i] = dt * index_l4   
                if len(np.where(level_max[i:]<3)[0]):
                    temp_a4u = np.where(level_max[i:]<3)[0]
                    TTA4U[i] = dt * temp_a4u[min(np.where(temp_a4u>index_l4)[0])]
            else:
                TTA4U[i] = np.nan
            
        # L3
        elif MODE[auto_level[i]] == "L3":
            print('automation mode: L3')
            TTA2F[i]  = 0            
            TTA3F[i]  = 0
            index_l2u = min(np.where(level_max[i:]<1)[0])
            TTA2U[i]  = dt * index_l2u
            index_l3u = min(np.where(level_max[i:]<2)[0])
            TTA3U[i]  = dt * index_l3u  

            if len(np.where(level_max[i:]==3)[0]):
                index_l4 = min(np.where(level_max[i:]==3)[0])
                TTA4F[i] = dt * index_l4  
                if len(np.where(level_max[i:]<3)[0]):
                    temp_a4u = np.where(level_max[i:]<3)[0]
                    TTA4U[i] = dt * temp_a4u[min(np.where(temp_a4u>index_l4)[0])]
            else:
                TTA4U[i] = np.nan

        # L4
        elif MODE[auto_level[i]] == "L4":
            print('automation mode: L4')
            TTA2F[i] = 0            
            TTA3F[i] = 0            
            TTA4F[i] = 0 
            index_l2u = min(np.where(level_max[i:]<1)[0])
            TTA2U[i]  = dt * index_l2u
            index_l3u = min(np.where(level_max[i:]<2)[0])
            TTA3U[i]  = dt * index_l3u  
            index_l4u = min(np.where(level_max[i:]<3)[0])
            TTA4U[i]  = dt * index_l4u  
        
        # summary      
        print('TTA2F',TTA2F[i])  
        print('TTA3F',TTA3F[i])  
        print('TTA4F',TTA4F[i])       
        print('TTA2U',TTA2U[i])  
        print('TTA3U',TTA3U[i])  
        print('TTA4U',TTA4U[i])  


    return TTA2F,TTA3F,TTA4F,TTA2U,TTA3U,TTA4U


                  

if __name__ == '__main__':
    '''
    Parameters
    ----------
    + seq_len: int
        length of the sequence

    + level_max: np.array
        maximum automation level

    + auto_level: np.array
        level of automation mode   

    + dt: float
        time of each step
    '''

    seq_len = 80
    dt = 1
    
    a1 = 0 * np.ones((10))
    a2 = 2 * np.ones((10))
    a3 = 3 * np.ones((10))
    a4 = 1 * np.ones((10))
    a5 = 2 * np.ones((10))
    a6 = 3 * np.ones((10)) 
    a7 = 2 * np.ones((10)) 
    a8 = 0 * np.ones((10)) 
    temp1 = np.hstack((np.hstack((a1,a2)),a3))
    temp2 = np.hstack((temp1,a4))
    temp3 = np.hstack((temp2,a5))
    temp4 = np.hstack((temp3,a6))
    temp5 = np.hstack((temp4,a7))

    level_max = np.hstack((temp5,a8))  

    b1 = 0 * np.ones((10))
    b2 = 1 * np.ones((10))
    b3 = 2 * np.ones((10))
    b4 = 1 * np.ones((10))
    b5 = 2 * np.ones((10))
    b6 = 3 * np.ones((10)) 
    b7 = 2 * np.ones((10)) 
    b8 = 0 * np.ones((10)) 
    temp11 = np.hstack((np.hstack((b1,b2)),b3))
    temp21 = np.hstack((temp11,b4))
    temp31 = np.hstack((temp21,b5))
    temp41 = np.hstack((temp31,b6))
    temp51 = np.hstack((temp41,b7))
    auto_level = np.hstack((temp51,b8))       

    TTA2F,TTA3F,TTA4F,TTA2U,TTA3U,TTA4U = compute_ttax(seq_len, auto_level, level_max, dt)
    
    

    plt.figure(figsize=(8,6))
    plt.subplot(221)
    plt.plot(level_max,'r.')
    plt.ylabel('level max')
    plt.xlabel('time')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)
    plt.subplot(223)
    plt.plot(auto_level,'g.')
    plt.ylabel('auto level')
    plt.xlabel('time')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)
    plt.subplot(222)
    plt.plot(np.log10(TTA2F+0.01*np.ones((seq_len,1))),'b.')
    plt.axis([0, 80, -3, 5])
    plt.ylabel('log(TTA2F)')
    plt.xlabel('time')
    plt.grid(True)
    plt.subplot(224)
    plt.plot(np.log10(TTA2U+0.01*np.ones((seq_len,1))),'r.')
    plt.axis([0, 80, -3, 5])
    plt.ylabel('log(TTA2U)')
    plt.xlabel('time')
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(8,6))
    plt.subplot(221)
    plt.plot(level_max,'r.')
    plt.ylabel('level max')
    plt.xlabel('time')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)
    plt.subplot(223)
    plt.plot(auto_level,'g.')
    plt.ylabel('auto level')
    plt.xlabel('time')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)
    plt.subplot(222)
    plt.plot(np.log10(TTA3F+0.01*np.ones((seq_len,1))),'b.')
    plt.axis([0, 80, -3, 5])
    plt.ylabel('log(TTA3F)')
    plt.xlabel('time')
    plt.grid(True)
    plt.subplot(224)
    plt.plot(np.log10(TTA3U+0.01*np.ones((seq_len,1))),'r.')
    plt.axis([0, 80, -3, 5])
    plt.ylabel('log(TTA3U)')
    plt.xlabel('time')
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(8,6))
    plt.subplot(221)
    plt.plot(level_max,'r.')
    plt.ylabel('level max')
    plt.xlabel('time')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)
    plt.subplot(223)
    plt.plot(auto_level,'g.')
    plt.ylabel('auto level')
    plt.xlabel('time')
    plt.yticks([0, 1, 2, 3], ['L0','L2','L3','L4'])  
    plt.axis([0, 80, -0.01, 3])
    plt.grid(True)
    plt.subplot(222)
    plt.plot(np.log10(TTA4F+0.01*np.ones((seq_len,1))),'b.')
    plt.axis([0, 80, -3, 5])
    plt.ylabel('log(TTA4F)')
    plt.xlabel('time')
    plt.grid(True)
    plt.subplot(224)
    plt.plot(np.log10(TTA4U+0.01*np.ones((seq_len,1))),'r.')
    plt.axis([0, 80, -3, 5])
    plt.ylabel('log(TTA4U)')
    plt.xlabel('time')
    plt.grid(True)
    plt.show()