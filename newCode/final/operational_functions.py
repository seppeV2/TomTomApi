from math import sqrt
import pathlib 
import numpy as np
import pandas as pd
from sympy import *
from scipy.stats import linregress 
import matplotlib.pyplot as plt
import seaborn as sns
from dyntapy.demand_data import od_matrix_from_dataframes
import geopandas as gpd
import os
from math import ceil, floor


# This function sets up the test cases we work with (to start)
# An average work day flow is taken from the morning peak hour (7 am to 9 am)
def setup_test_case(nameZoning: str, nameTomTomCsv: str):
    #first retrieve the original od matrix
    #get the right parameters
    od_table = pd.read_csv(str(pathlib.Path(__file__).parents[1])+"/data/Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.csv")
    zoning = gpd.read_file(str(pathlib.Path(__file__).parents[2])+'/AllZonings/{}.zip'.format(nameZoning))
    origin_column = 'H'
    destination_column = 'B'
    zone_column = 'ZONENUMMER'
    flow_column = 'Tab:8'
    #result from 7 to 8
    result1  = od_matrix_from_dataframes(od_table,zoning, origin_column, destination_column,zone_column, flow_column)
    
    
    #only get the first result from the previous function
    od8 = result1[0]
    flow2_column = 'Tab:9'
    #result form 8 to 9
    result2 = od_matrix_from_dataframes(od_table, zoning, origin_column, destination_column, zone_column, flow2_column)
    od9 = result2[0]
    #result form 7 to 9
    original_od = np.add(od8,od9)
    path = str(pathlib.Path(__file__).parent) + '/OD_Matrices/'
    pd.DataFrame(original_od).to_csv(path+'OD_MOW_{}.csv'.format(nameZoning), header = False, index = False)

    #now retrieve the data from tomtom (saved as cvs file in data/move_results)
    path = str(pathlib.Path(__file__).parents[1])+'/data/move_results/{}.csv'.format(nameTomTomCsv)
        #this needs to change for every new tomtom move analysis 
    flow_column = 'Date range: 2021-01-25 - 2021-01-29 Time range: 07:00 - 09:00'
    tomtom_od = od_matrix_from_tomtom(path, flow_column)
    #we want the average flow in the peak hours so we divide the flow from one work week by 5
    tomtom_od = np.round((tomtom_od / 5),3)
    path = str(pathlib.Path(__file__).parent) + '/OD_Matrices/'
    pd.DataFrame(tomtom_od).to_csv(path+'OD_TOM_{}.csv'.format(nameZoning), header = False, index = False)
    return original_od, tomtom_od

# function to make a np.array from the tomtom move csv file
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

    return od_matrix[0:size-1,0:size-1]

# import the csv files (faster than loaden via setup test case)
def import_test_case(nameZoning: str):
    path = str(pathlib.Path(__file__).parent) + '/OD_Matrices/'
    original_od = pd.read_csv(path+'OD_MOW_{}.csv'.format(nameZoning), header = None).to_numpy()
    tomtom_od = pd.read_csv(path+'OD_TOM_{}.csv'.format(nameZoning), header = None).to_numpy()

    return original_od,tomtom_od

# function to calculate the gap between two matrices (of the same shape)
# In this case we use RMSE to find that gap between every OD pair
# Sum of all these together form the gap.
def calculate_gap_RMSE(matrix1, matrix2):
    gap = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix1)):
            # we don't care about the intra zonal traffic
            if i != j:
                gap += sqrt((1/matrix1.size)*(matrix1[i, j] - matrix2[i, j])**2)
    return gap

# function to make a list form the matrix 
    # input = squared matrix and optional the shape matrix (so only relevant elements in the list)
    # returns the list  
def matrix_to_list(matrix, shape = []):
    if shape == []:
        return np.array(matrix.reshape([len(matrix)**2,]))
    else:
        list = []
        for i in range(len(shape)):
            for j in range(len(shape)):
                if shape[i,j] == 1:
                    list.append(matrix[i,j])
        return np.array(list)
        

# function to make a matrix from the list
    # input = list (made of squared matrix) + shape matrix (only put back the relevant elements)
    # return = resized matrix
def list_to_matrix(list, shapeMatrix=[]):
    if shapeMatrix != []:
        matrix  = np.zeros(shapeMatrix.shape)
        counter = 0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if shapeMatrix[i,j] == 1:
                    matrix[i,j] += list[counter]
                    counter+=1
        return matrix
    else:
        list_bis = np.array(list)
        return list_bis.reshape([sqrt(len(list)), sqrt(len(list))])

# get a split on with a fixed size jump 
def get_split_fixed(matrix, fixed_size = 0, cutoffs = []):

    max = ceil(np.max(matrix))
    
    extra = 0 if (fixed_size ==0 or max%fixed_size == 0) else 1
    slices = int((max//fixed_size) + extra) if fixed_size != 0 else len(cutoffs)-1
    cutoffs = [x*fixed_size for x in range(slices + 1)] if cutoffs == [] else cutoffs
    #make sure everything is in the range
    cutoffs.insert(0, -10**10)
    cutoffs[-1] += 1
    ranges = []
    shapes = []
    for k in range(slices+1):
        shape = np.zeros(matrix.shape)
        lower = cutoffs[k]
        upper = cutoffs[k+1]
        
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if lower <= matrix[i,j] < upper:
                    shape[i,j] += 1

        if np.sum(shape) != 0:
            shapes.append(shape)
            ranges.append((lower,upper))
    return shapes, ranges

    # split matrix in shapes according the values inside the matrix
def get_split_matrices(matrix, slices: int = -1, cutoffs = []):
    #split matrices in x slices between min (0) and max values
    if slices != -1:
        shapes = []
        ranges = []
        prev = -1
        for k in range(slices):
            if cutoffs == []:
                next = np.max(matrix)*((k+1)/slices)
            else:
                if k == slices-1:
                    next = np.max(matrix)+1
                else:
                    next = cutoffs[k]
            shape = np.zeros(matrix.shape)
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    
                    if prev < matrix[i,j] <= next:
                        shape[i,j] += 1
            ranges.append((prev, next))
            prev = next
            shapes.append(shape)
            
        return shapes, ranges
    #split matrix in two slices (before and after average)
    elif slices == 1:
        return np.full((len(matrix), len(matrix)), 1)
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
