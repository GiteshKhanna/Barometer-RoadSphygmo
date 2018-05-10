import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

#Setting Parameters
#winSize in seconds
#compDistance in seconds
#jumpThreshold in meters
##########

winSize=10
mob = 's3'
folder='vehicle/raw_csv'
file_name = 's3_vehicle_7_bandra'
ext = '.csv'




#Importing dataset
dataset = pd.read_csv(mob+'/'+folder+'/'+file_name+ext)
jumps= dataset.iloc[: ,6].values

#Trimming data to fit to windowSize
rem=jumps.shape[0]%winSize
dataSize= jumps.shape[0] - rem
jumps_restructured= jumps[:dataSize]


#Creating input data
jumps_restructured= jumps_restructured.reshape((int(dataSize/winSize),winSize))


print(jumps_restructured.shape)

d = {}
for i in range(winSize):
    d['AtmPressure#'+str(i)] = jumps_restructured[:,i]

d['Label'] = np.ones(jumps_restructured.shape[0])
df = pd.DataFrame(d)
df.to_csv(file_name+'_Features'+ext)
