from operational_functions import heatmaps, calculate_gap_RMSE,matrix_to_list, normalize, get_split_matrices, list_to_matrix, visualize_splits, setup_test_case, get_split_fixed
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
def approx_linear(original_od, tomtom_od, explanatory = [], list_return = False):
    # If explanatory is empty this means that we perform a simple regression
    if explanatory == []:
        if not list_return:
            X = matrix_to_list(tomtom_od).reshape((-1, 1))
            Y = matrix_to_list(original_od)
        elif list_return:
            X = tomtom_od.reshape((-1, 1))
            Y = original_od
            # Perform the model
        model = LinearRegression().fit(X, Y)
            # Get the prediction
        approx_list = model.predict(X)
        if not list_return:
            approx_matrix = list_to_matrix(approx_list)
            for i in range(len(original_od)):
                approx_matrix[i,i] = 0
            return approx_matrix, np.round(model.coef_,3), round(model.intercept_,3)
        elif list_return:
            return approx_list, np.round(model.coef_,3), round(model.intercept_,3)
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
def simple_linear_reg(original_od, tomtom_od, zone, norm = False, ):
    if norm:
        approx_od, slope, intercept = approx_linear(original_od/np.sum(original_od), tomtom_od/np.sum(tomtom_od)) 
        approx_od = approx_od * np.sum(tomtom_od)
    else:
        approx_od, slope, intercept = approx_linear(original_od, tomtom_od) 
    approx_gap = calculate_gap_RMSE(original_od, approx_od)
    approx_od_norm = np.max(approx_od)

    # Values to normalize the matrices
    original_od_norm = np.max(original_od)
    tomtom_od_norm = np.max(tomtom_od)

    if norm:
        adi = '/norm'
    else:
        adi = ''
    path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/heatmaps/simple_reg{}'.format(adi)
    os.makedirs(path, exist_ok=True)

    heatmaps(original_od, tomtom_od, zone, 'original OD matrix', 'tomtom OD matrix', '{} * [nxn] + {}'.format(slope,intercept), 'oriVStom_{}'.format(zone), path)
    heatmaps(original_od/original_od_norm, tomtom_od/tomtom_od_norm, zone, 'original OD matrix', 'tomtom OD matrix', '{} * [nxn] + {}'.format(slope,intercept), 'oriVStom_{}_normalized'.format(zone), path )
    return approx_gap, slope, intercept

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

    return approx_gap, slope, intercept

# Perform linear regression using clustering according network_proeprties
def split_linear_regression(original_od, tomtom_od, zone,network_property = '', method = '', fixed = True, fixed_jump = 0, cutoffs = []):
    path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/split_analysis/{}'.format(zone)
    os.makedirs(path, exist_ok=True)

    explanatory_od = create_OD_from_info(network_property+'_'+zone+'_dictionary', method) if  network_property != '' else None
    
    if not fixed:
        shapes = get_split_matrices(explanatory_od,3, cutoffs = [4000,8000])
        visualize_splits(shapes, zone ,path)
    else: 
        network_property
        shapes = get_split_fixed(original_od, fixed_size = fixed_jump) if cutoffs == [] else get_split_fixed(tomtom_od, _cutoffs = cutoffs)
        visualize_splits(shapes, zone ,path)
    approx_matrix = np.zeros(original_od.shape)
    slopes = []
    intercepts = []
    summary = ''
    for idx, shape in enumerate(shapes):
        orig_list = matrix_to_list(original_od, shape)
        tomtom_list = matrix_to_list(tomtom_od, shape)

        
        if orig_list != [] and tomtom_list != []:
            approx_list , slope, intercept = approx_linear(orig_list, tomtom_list, list_return = True) 
            slopes.append(slope)
            intercepts.append(intercept)
            approx_matrix += list_to_matrix(approx_list, shape)

            summary += 'Split {} with equations: Y = {} * X + {}\n'.format(idx, np.round(slope[0],3), np.round(intercept,3))


    #plot the equations
    x = np.linspace(0,10,300)
    _,ax = plt.subplots()
    for j in range(len(slopes)):    
        y1 = slopes[j]*x + intercepts[j]
        ax.plot(x,y1, label = 'Eq for {}th split of OD {}'.format(j+1,zone))
    ax.legend()
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_title('Linear equations from the {}th split of the matrix'.format(''))
    plt.savefig(path+'/linear_equations_split_{}.png'.format(zone))

    approx_gap = calculate_gap_RMSE(original_od, approx_matrix)
    return approx_gap, 'split on {}'.format(network_property), summary, shapes, approx_matrix

# linear regression with the residuals and network property 
def linear_residua(original_od, tomtom_od, network_property, zone):
    residua = original_od - tomtom_od
    explanatory_od = create_OD_from_info(network_property+'_'+zone+'_dictionary')
    explanatory_od_norm = explanatory_od

    intermediate_od, slope, intercept = approx_linear(residua, explanatory_od_norm)
    approx_od = intermediate_od + tomtom_od

    approx_gap = calculate_gap_RMSE(original_od, approx_od)

    path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/residua/heatmaps/residua_reg_{}'.format(network_property)
    os.makedirs(path, exist_ok=True)

    heatmaps(original_od, approx_od, zone, 'original OD matrix', 'approx OD matrix', '{} * [nxn] + {}'.format(slope,intercept), fileName = 'oriVSaprox_{}_{}'.format(zone,network_property), path=path)

    return approx_gap, slope, intercept, approx_od

def linear_residua_split(original_od, tomtom_od, network_property, zone, move):

    original_od, tomtom_od = setup_test_case(zone, move)
    residua = original_od - tomtom_od
    explanatory_od = create_OD_from_info(network_property+'_'+zone+'_dictionary')
    _, string, summary_split, shapes, intermediate = split_linear_regression(residua, explanatory_od, zone, fixed = True, fixed_jump = 10)
    approx_od = intermediate + tomtom_od
    approx_gap = calculate_gap_RMSE(approx_od, original_od)
    
    return approx_gap, string, summary_split, shapes