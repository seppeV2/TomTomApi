from math import sqrt
import pathlib 
import numpy as np
import pandas as pd

def od_matrix_from_move_csv(pathToFile):
    result = pd.read_csv(pathToFile)
    matrix = []
    for i in range(int(sqrt(len(result)))):
        row = []
        count = 0
        while result['Origin'][i*44+count] == ('Region '+str(i+1)):
            row.append(result['Date range: 2021-01-25 - 2021-01-31 Time range: 00:00 - 00:00'][(44*i)+count])
            count +=1
            if i*44+count == len(result)-1:
                break

        matrix.append(row)
        
    return matrix

