import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

#Setting Parameters
#winSize in seconds
#compDistance in seconds
#jumpThreshold in meters
##########
winSize = 20
compDistance = 5
jumpThreshold = 1

mob = 'nexus_4'
act_state = 'train'
file_name = 'nexus4_vehicle'
ext = '.csv'

windows = np.array([])
jumps = np.array([])



#Importing dataset
dataset = pd.read_csv(mob+'/'+act_state+'/'+file_name+ext)
alt= dataset.iloc[: ,0].values
atm_press = deepcopy(alt)
alt = 44330 * (1-np.power((atm_press/101325),(1/5.255)))

#Trimming Data to fit windowsize
print('Data Fetched size:'+ str(alt.shape[0]))
dataSize = alt.shape[0] - (alt.shape[0]-compDistance)%winSize
alt_restructured= alt[compDistance:dataSize]
print('Restructured data size: ' +str(alt_restructured.shape[0]))



#Creating windows
#windows-> Contains datapoints of all the windows in an array
#dissectedWindows-> contains a matrix with rows -> all windows, columns->datapoints in every window
############
for i in range(0,len(alt_restructured)-winSize+1):
    windows = np.append(windows,alt_restructured[i:winSize+i])
    '''print(len(windows))'''

dissectedWindows = windows.reshape(len(alt_restructured)-winSize+1,winSize)


#Finding Jumps
lateTimestamp = deepcopy(windows)
earlyTimestamp =  alt[0:compDistance]
earlyTimestamp = np.append(earlyTimestamp,windows[0:len(windows)-compDistance])
jumps = np.abs(lateTimestamp - earlyTimestamp)
jumps[jumps > jumpThreshold] = 1
jumps[jumps != 1] = 0


#Counting jumps in every Window
windowJumps = jumps.reshape(dissectedWindows.shape)
windowJumps = np.sum(windowJumps,axis = 1)
print('windowJumps Size:'+str(windowJumps.shape[0]))
print(windowJumps)

#Storing in excel file
df = pd.DataFrame({'Jumps':windowJumps.tolist()})
df.to_csv(file_name+'_jumps'+ext)

#creating plot
plt.plot(np.arange(windowJumps.shape[0]),windowJumps)
plt.title(act_state+file_name)
plt.xlabel('Window Time')
plt.ylabel("Jumps")
plt.savefig(file_name)
plt.show()
