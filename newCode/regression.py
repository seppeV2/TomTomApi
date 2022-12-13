from operational_functions import heatmaps, calculate_gap_RMSE,matrix_to_list, normalize, get_split_matrices, list_to_matrix, visualize_splits, setup_test_case
from output_functions import bars, equations, scatters, calculate_model, correlation_analyses, gap_bars
import pathlib
import pandas as pd 
import numpy as np
from scipy.stats import pearsonr
from data_processing_properties import create_OD_from_info
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from matplotlib.container import BarContainer
np.set_printoptions(suppress=True)

# function to calculate a approx of the mow matrix via linear regression
# return = the approx matrix
def approx_linear(original_od, tomtom_od, explanatory = []):
    # If explanatory is empty this means that we perform a simple regression
    if explanatory == []:
        X = matrix_to_list(tomtom_od).reshape((-1, 1))
        Y = matrix_to_list(original_od)
            # Perform the model
        model = LinearRegression().fit(X, Y)
            # Get the prediction
        approx_list = model.predict(X)
        approx_matrix = list_to_matrix(approx_list)
        for i in range(len(original_od)):
            approx_matrix[i,i] = 0
        return approx_matrix, np.round(model.coef_,3), round(model.intercept_,3)
    else:
        # Explanatory variables are stored in a matrix these are put together and shaped correctly
        X = np.transpose([matrix_to_list(tomtom_od), matrix_to_list(explanatory)])
        Y = matrix_to_list(original_od)
            # Perform the Model
        model = LinearRegression().fit(X, Y)
            # Get the prediction
        approx_list = model.predict(X)
        approx_matrix = list_to_matrix(approx_list)
        for i in range(len(original_od)):
            approx_matrix[i,i] = 0
                # model.coef_ = list with the slopes for each matrix (tomtom + explanatory)
        return approx_matrix, np.round(model.coef_,3), round(model.intercept_,3)



# Perform a simple linear regression between two matrices
def simple_linear_reg(original_od, tomtom_od, zone):
    approx_od, slope, intercept = approx_linear(original_od, tomtom_od) 
    approx_gap = calculate_gap_RMSE(original_od, approx_od)
    approx_od_norm = np.max(approx_od)

    # Values to normalize the matrices
    original_od_norm = np.max(original_od)
    tomtom_od_norm = np.max(tomtom_od)

    path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/heatmaps/simple_reg'
    os.makedirs(path, exist_ok=True)

    heatmaps(original_od, tomtom_od, zone, 'original OD matrix', 'tomtom OD matrix', 'Original vs TomTom (normalized)', 'oriVStom_{}'.format(zone), path)
    heatmaps(original_od/original_od_norm, tomtom_od/tomtom_od_norm, zone, 'original OD matrix', 'tomtom OD matrix', 'Original vs TomTom (normalized)', 'oriVStom_{}_normalized'.format(zone), path )
    return approx_gap

# Perform linear regression with explanatory variables
def explanatory_linear_regression(original_od, tomtom_od, zone, network_property, converge_method):

    # Retrieve the wright property od matrix (using the given method)
    explanatory_od = create_OD_from_info(network_property+'_'+zone+'_dictionary', converge_method)
    
    # Values to normalize the matrices
    explanatory_od_norm = np.max(explanatory_od)
    original_od_norm = np.max(original_od)
    

    # Perform the model
    approx_od, slope, intercept = approx_linear(original_od, tomtom_od, explanatory_od/explanatory_od_norm) 
    approx_gap = calculate_gap_RMSE(original_od, approx_od)
    approx_od_norm = np.max(approx_od)
    
    # Make the equation string 
    linRegString = '{} * [nxn]'.format(slope[0])
    for i in range(1,len(slope)):
        linRegString += ' + {} * [nxn] '.format(slope[i])
    linRegString += '+ '+str(intercept)

    # Make the heatmaps
    path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/heatmaps/{}'.format(network_property)
    os.makedirs(path, exist_ok=True)
    heatmaps(original_od, approx_od, zone, 'original OD matrix', 'approx OD matrix', '{}'.format(linRegString), 'oriVSaprox_{}'.format(zone), path)
    heatmaps(original_od/original_od_norm, approx_od/approx_od_norm, zone, 'original OD matrix', 'approx OD matrix', '{} (norm)'.format(linRegString, intercept), 'oriVSaprox_{}_normalized'.format(zone), path)

    return approx_gap

