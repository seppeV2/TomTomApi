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
from sklearn.linear_model import LinearRegression 
from math import ceil, floor

def heatmaps(matrix1, matrix2, zone, name1, name2, addiTitle='' , fileName='', path = ''):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('{} {}'.format(zone, addiTitle))
    sns.heatmap(matrix2, ax=ax[0]).set(title=name2)
    sns.heatmap(matrix1, ax=ax[1]).set(title=name1)
    if path == '':
        path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/heatmaps'
        os.makedirs(path, exist_ok=True)
    if fileName != '':
        plt.savefig(path+'/heatmap_{}.png'.format(fileName))
    else: 
        plt.savefig(path+'/heatmap_{}.png'.format(zone))
    plt.close()

def heatmap(matrix, zone, name, addiTitle='' , fileName='', path = ''):
    fig, ax = plt.subplots()
    fig.suptitle('{} {}'.format(zone, addiTitle))
    sns.heatmap(matrix, ax=ax).set(title=name)
    if path == '':
        path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/heatmaps'
        os.makedirs(path, exist_ok=True)
    if fileName != '':
        plt.savefig(path+'/heatmap_{}.png'.format(fileName))
    else: 
        plt.savefig(path+'/heatmap_{}.png'.format(zone))
    plt.close()

   
def visualize_splits(shapes, zone, path):
    amount_shapes = len(shapes)
    extra = 0 if len(shapes)%2 == 0 else 1
    for i in range(int(amount_shapes//2)+extra):
        fig, ax = plt.subplots(1,2)
        fig.suptitle(zone)
        sns.heatmap(shapes[i*2],ax=ax[0]).set(title='shape_{}'.format(i*2+1))
        if (i*2)+1 < len(shapes):
            sns.heatmap(shapes[i*2+1],ax=ax[1]).set(title='shape_{}'.format(i*2+2))
        plt.savefig(path+'/visual_shapes_{}_{}.png'.format(zone, i+1))  
        plt.close()
    
def outliers(matrix, zone):
    fig,ax = plt.subplots()
    ax.set_title('Normalized difference {}'.format(zone))
    sns.heatmap(matrix)
    plt.savefig(str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/outliers_{}.png'.format(zone))

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

# function to find the coef where the squared error is the least significant
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

# return the normalized matrix
def normalize(matrix):
    return matrix/(np.sum(matrix)), np.sum(matrix)

# split matrix in shapes according the values inside the matrix
def get_split_matrices(matrix, slices: int = -1, cutoffs = []):
    #split matrices in x slices between min (0) and max values
    if slices != -1:
        shapes = []
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
            prev = next
            shapes.append(shape)
        return shapes
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

# get a split on with a fixed size jump 
def get_split_fixed(matrix, fixed_size = 0, _cutoffs = []):

    min = floor(np.min(matrix))
    max = ceil(np.max(matrix))
    
    extra = 0 if (fixed_size ==0 or max%fixed_size == 0) else 1
    slices = int((max//fixed_size) + extra) if fixed_size != 0 else len(_cutoffs)
    cutoffs = [x*fixed_size + min for x in range(slices + 1)] if _cutoffs == [] else _cutoffs

    shapes = []
    ones = 0
    for k in range(slices):
        shape = np.zeros(matrix.shape)
        lower = cutoffs[k]
        upper = cutoffs[k+1]
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if lower <= matrix[i,j] < upper:
                    shape[i,j] += 1
        shapes.append(shape)
        ones += np.sum(shape)
    return shapes

# This function sets up the test cases we work with (to start)
# An average work day flow is taken from the morning peak hour (7 am to 9 am)
def setup_test_case(nameZoning: str, nameTomTomCsv: str):
    #first retrieve the original od matrix
    #get the right parameters
    od_table = pd.read_csv(str(pathlib.Path(__file__).parent)+"/data/Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.csv")
    zoning = gpd.read_file(str(pathlib.Path(__file__).parents[1])+'/AllZonings/{}.zip'.format(nameZoning))
    origin_column = 'H'
    destination_column = 'B'
    zone_column = 'ZONENUMMER'
    flow_column = 'Tab:8'
    #result from 7 to 8
    result1  = od_matrix_from_dataframes(od_table,zoning, origin_column, destination_column,zone_column, flow_column)
    #only get the first result form the previous function
    od8 = result1[0]
    flow2_column = 'Tab:9'
    #result form 8 to 9
    result2 = od_matrix_from_dataframes(od_table, zoning, origin_column, destination_column, zone_column, flow2_column)
    od9 = result2[0]
    #result form 7 to 9
    original_od = np.add(od8,od9)

    #now retrieve the data from tomtom (saved as cvs file in data/move_results)
    path = str(pathlib.Path(__file__).parent)+'/data/move_results/{}.csv'.format(nameTomTomCsv)
        #this needs to change for every new tomtom move analysis 
    flow_column = 'Date range: 2021-01-25 - 2021-01-29 Time range: 07:00 - 09:00'
    tomtom_od = od_matrix_from_tomtom(path, flow_column)
    #we want the average flow in the peak hours so we divide the flow from one work week by 5
    tomtom_od = tomtom_od / 5
    return original_od, tomtom_od

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

def calculate_perc_matrix(matrix1, matrix2, zone):
    full_gap = calculate_gap_RMSE(matrix1, matrix2)
    perc_matrix = np.zeros(matrix1.shape)
    for i in range(len(matrix1)):
        for j in range(len(matrix1)):
            # we don't care about the intra zonal traffic
            if i != j:
                perc_matrix[i,j] += (sqrt((1/matrix1.size)*(matrix1[i, j] - matrix2[i, j])**2))/full_gap

    fig, ax = plt.subplots()
    fig.suptitle('matrix percentage {}'.format(zone))
    sns.heatmap(perc_matrix, ax=ax)
    path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/heatmaps/percentage_matrix'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+'/heatmap_gap_percentage{}.png'.format(zone))
    plt.close()
    
    
    return perc_matrix
