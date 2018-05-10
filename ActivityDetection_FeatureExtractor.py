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
mob = 'nexus4_vehicle'
file_name = 'nexus4_vehicle'
key = '_jumps'
ext = '.csv'




#Importing dataset
dataset = pd.read_csv(mob+'/'+file_name+key+ext)
jumps= dataset.iloc[: ,1].values

#Trimming data to fit to windowSize
rem=jumps.shape[0]%winSize
dataSize= jumps.shape[0] - rem
jumps=restructured= jumps[:dataSize]

#Creating windows
jumps_restructured= jumps.reshape((int(dataSize/winSize),winSize))


#Saving into an excel file
d = {}
for i in range(winSize):
    d['Jumps#'+str(i)] = jumps_restructured[:,i]

df = pd.DataFrame(d)
df.to_csv(file_name+'_Features'+ext)
