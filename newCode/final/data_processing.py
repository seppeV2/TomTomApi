import pandas as pd
import pathlib
import numpy as np

# From the properties csv file (zone number and property information)
# An OD file is made according the different method (that is a variable of the function)
# Sum = (O + D), Average = (O + D)/2, Destination (based) = (D) (with O, D = property value of origin and destination)
def create_OD_from_info(fileName, mergeWay= 'sum', landUse = False):
    # Head of the data pandas frame
    dataHead = ['zoneName', 'amount']
    data = pd.read_csv(str(pathlib.Path(__file__).parent)+'/Info_properties/statbel/{}.csv'.format(fileName), names = dataHead) if not landUse\
        else pd.read_csv(str(pathlib.Path(__file__).parent)+'/Info_properties/poidpy/{}.csv'.format(fileName), names = dataHead)
    OD = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                if mergeWay == 'sum':
                    OD[i,j] += data['amount'][i] + data['amount'][j]
                elif mergeWay == 'average':
                    OD[i,j] += (data['amount'][i] + data['amount'][j])/2
                elif mergeWay == 'destination':
                    OD[i,j] += data['amount'][j]

                elif mergeWay == 'origin':
                    OD[i,j] += data['amount'][j]
    return OD