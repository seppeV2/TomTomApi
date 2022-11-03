from math import sqrt
import pathlib 
import numpy as np
import pandas as pd
from sympy import *
from scipy.optimize import minimize 



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

#function to find the coef where the squared error is the least significant
def find_optimal_coef(array1,array2):
    
    X = Symbol('X')
    squaredError = 0
    for i in range(len(array1)):
        for j in range(len(array1)):
            if j != i:
                squaredError += (array1[i][j] - X*array2[i][j])**2

    #now minimize this function
    squaredErrorPrime = squaredError.diff(X)
    return solve(squaredErrorPrime, X)
    
#function to calculate the gap between two od's from the same size
def calculate_gap(matrix1, matrix2):
    gap = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1)):
            if i != j:
                gap += abs(matrix1[i][j] - matrix2[i][j])
    
    return gap