from math import sqrt
import pathlib 
import numpy as np
import pandas as pd
from sympy import *
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import seaborn as sns


def heatmaps(od_original, od_tomtom):
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(od_tomtom, ax=ax[0]).set(title='TomTom')
    sns.heatmap(od_original, ax=ax[1]).set(title='Original')
    plt.show()


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
    #input = squared matrix and optional the shape matrix (so only relevant elements in the list)
    #returns the list  
def matrix_to_list(matrix, shape = []):
    if shape == []:
        return matrix.reshape([1,len(matrix)**2])
    else:
        list = []
        for i in range(len(shape)):
            for j in range(len(shape)):
                if shape[i,j] == 1:
                    list.append(matrix[i,j])
        return np.array(list)

#function to make a matrix from the list
    #input = list (made of squared matrix) + shape matrix (only put back the relevant elements)
    #return = resized matrix
def list_to_matrix(list, shapeMatrix):
    matrix  = np.zeros(shapeMatrix.shape)
    counter = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if shapeMatrix[i,j] == 1:
                matrix[i,j] += list[counter]
                counter+=1
    return matrix

#return the normalized matrix
def normalize(matrix):
    print(np.sum(matrix))
    return matrix/(np.sum(matrix))

#split matrix in shapes according the values inside the matrix
def get_split_matrices(matrix, slices: int = -1):
    #split matrices in x slices between min (0) and max values
    if slices != -1:
        shapes = []
        prev = -1
        for k in range(slices):
            next = np.max(matrix)*((k+1)/slices)
            shape = np.zeros(matrix.shape)
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if prev < matrix[i,j] <= next:
                        shape[i,j] += 1
            prev = next
            shapes.append(shape)
        return shapes
    #split matrix in two slices (before and after average)
    else: 
        shape1 = np.zeros(matrix.shape)
        shape2 = np.zeros(matrix.shape)
        average = np.average(matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i,j] <= average:
                    shape1[i,j] = 1
                else:
                    shape2[i,j] = 1 
        return [shape1, shape2]

    

