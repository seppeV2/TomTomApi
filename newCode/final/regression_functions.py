from operational_functions import matrix_to_list, list_to_matrix, calculate_gap_RMSE, get_split_fixed
from output_functions import heatmaps,correlation_analyses
import pandas as pd 
import numpy as np
from data_processing import create_OD_from_info
from sklearn.linear_model import LinearRegression

# function to calculate a approx of the mow matrix via linear regression
# return = the approx matrix
def approx_linear(original_od, tomtom_od, list_return = False):
    # If explanatory is empty this means that we perform a simple regression
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
        return approx_matrix, round(model.coef_[0],3), round(model.intercept_,3)
    elif list_return:
        return approx_list, round(model.coef_[0],3), round(model.intercept_,3)


# Perform a simple linear regression between two matrices
def simple_linear_reg(original_od, tomtom_od ):

    approx_od, slope, intercept = approx_linear(original_od, tomtom_od) 
    approx_gap = calculate_gap_RMSE(original_od, approx_od)

    return approx_gap, slope, intercept



def linear_residua_split(original_od, tomtom_od, network_property, zone):

    residua = original_od - tomtom_od
    explanatory_od = create_OD_from_info(network_property+'_'+zone+'_dictionary')
    cutoffs = [-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1]
    intermediate, slopes, intercepts, ranges = split_linear_regression(residua, explanatory_od, fixed_jump = 10, residua = True, extra = tomtom_od)#, _cutoffs = cutoffs)
    approx_od = intermediate + tomtom_od
    approx_gap = calculate_gap_RMSE(approx_od, original_od)
    
    return approx_gap, slopes, intercepts, ranges

# Perform linear regression using clustering according network_proeprties
def split_linear_regression(matrix1, matrix2 ,fixed_jump = 0, _cutoffs = [] , residua = False, extra = []):

    if fixed_jump != 0 and _cutoffs == []:
        shapes, ranges = get_split_fixed(matrix1, fixed_size = fixed_jump) if not residua else get_split_fixed(matrix1, fixed_size = fixed_jump)
    elif _cutoffs != []:
        shapes, ranges = get_split_fixed(matrix1, cutoffs = _cutoffs) if not residua else get_split_fixed(matrix1, cutoffs = _cutoffs)
        
    approx_matrix = np.zeros(matrix1.shape)
    slopes = []
    intercepts = []
    for shape in shapes:
        list_1 = matrix_to_list(matrix1, shape)
        list_2 = matrix_to_list(extra, shape)

        if list_1 != [] and list_2 != []:
            approx_list , slope, intercept = approx_linear(list_1, list_2, list_return = True) 
            slopes.append(slope)
            intercepts.append(intercept)
            approx_matrix += list_to_matrix(approx_list, shape)

    return approx_matrix, slopes, intercepts, ranges

def land_use_split(original_od, tomtom_od, zone):
    land_use = create_OD_from_info('landuse_{}_dictionary'.format(zone), mergeWay='destination', landUse=True)
    shapes, ranges = get_split_fixed(land_use, fixed_size = 1)
    print(land_use)
    print(shapes)


def adjust_property(original_od, tomtom_od, network_property, zone):
    prop_od = create_OD_from_info('{}_{}_dictionary'.format(network_property ,zone), mergeWay='sum')
        #all values of norm_prop_od are between 0 and 1
    norm_prop_od = prop_od/(np.max(prop_od))

    shapes, ranges = get_split_fixed(norm_prop_od, fixed_size = 0.1)
    residua = original_od-tomtom_od

    slopes = []
    intercepts = []
    approx_matrix = np.zeros(residua.shape)
    for shape in shapes:
        residua_shape_list = matrix_to_list(residua, shape=shape)
        norm_prop_list = matrix_to_list(norm_prop_od, shape=shape)
        approx_list , slope, intercept = approx_linear(residua_shape_list, norm_prop_list, list_return = True) 
        slopes.append(slope)
        intercepts.append(intercept)
        approx_matrix += list_to_matrix(approx_list, shape)

    correlation_analyses(residua, norm_prop_od, network_property, zone, method = "sum", name='original')
    correlation_analyses(residua, approx_matrix, network_property, zone, method = "sum", name = 'regression')
    print('zone = {}'.format(zone))
    #print('approx matrix = {}'.format(approx_matrix))
    print('slopes = {}'.format(slopes))
    print('intercepts = {}'.format(intercepts))
    print('ranges = {}'.format(ranges))
    print('gap between residua and property approx = {}\n'.format(calculate_gap_RMSE(residua, approx_matrix)))

    return approx_matrix



