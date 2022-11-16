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
            if j != i:
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

#function to make a list form the matrix 
    #input = squared matrix
    #returns the list  
def matrix_to_list(matrix):
    return matrix.reshape([1,len(matrix)**2])

#function to make a matrix from the list
    #input = list (made of squared matrix) + len of that matrix
    #return = resized matrix
def list_to_matrix(list, size):
    return list.reshape([size,size])

def normalize(matrix):
    return matrix/(sum(matrix))

def get_split_matrices(matrix):
    shape1 = np.zeros(matrix.shape)
    tuple1 = np.zeros(matrix.shape,dtype = object)
    shape2 = np.zeros(matrix.shape)
    tuple2 = np.zeros(matrix.shape,dtype = object)
    average = np.average(matrix)

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i,j] <= average:
                tuple1[i,j] = (matrix[i,j], 1)
                tuple2[i,j] = (matrix[i,j], 0)
                shape1[i,j] = 1
            else:
                tuple1[i,j] = (matrix[i,j], 0)
                tuple2[i,j] = (matrix[i,j], 1)
                shape2[i,j] = 1 
    return [shape1, shape2] , [tuple1, tuple2]

#matrix 1 is a tuple matrix shaped with the get split funtion 
#matrix 2 is a normal matrix where you want the values at the same location
def matrix_to_list_splitsed(matrix1, matrix2):
    list1 = []
    list2 = []
    for i in range(len(matrix2)):
        for j in range(len(matrix2)):
            if matrix1[i,j][1] == 1:
                list1.append(matrix1[i,j][0])
                list2.append(matrix2[i,j])
    return list1, list2

