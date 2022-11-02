from math import sqrt
import pathlib 
import numpy as np
import pandas as pd

#function to make a np.array from the tomtom move csv file
def od_matrix_from_tomtom(pathToFile, flow):
    #read the csv file in as panda dataframe
    result = pd.read_csv(pathToFile)
    #size = sqrt of the lines = amount of zones in the tomtom analysis
    size = int(sqrt(len(result)))
    #create the empty array with zeros
    od_matrix = np.zeros((size, size), dtype=np.float64)
    #fill the empty array with the flows
    for i in range(size):
        for j in range(size):
            od_matrix[i][j] += int(result[flow][i*size+j])

    return od_matrix
