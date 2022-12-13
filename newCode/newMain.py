from operational_functions import heatmaps, calculate_gap_RMSE,matrix_to_list, normalize, get_split_matrices, list_to_matrix, visualize_splits, setup_test_case
from output_functions import bars, equations, scatters, calculate_model, correlation_analyses, gap_bars
from regression import simple_linear_reg, explanatory_linear_regression
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


def main():
    zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
    moves = ['LeuvenExternal', 'BruggeExternal', 'HasseltExternal']
    network_properties = ['population_statbel', 'households_statbel', 'cars_statbel']
    methods = ['sum', 'destination', 'origin']
    #decide which analysis you want to run
    run_correlation = False
    explanatory_analyses = True
    gaps_analysis =  True

    total_gap = {}
    
    if explanatory_analyses:
        for property in network_properties:
            total_gap[property] = {}
    else:
        total_gap['none'] = {}

    for zone, move in zip(zonings,moves):

        # Set up the matrices for this zone and move
        original_od, tomtom_od = setup_test_case(zone, move)

        # Calculate the original Gap (via RMSE)
        original_gap = calculate_gap_RMSE(original_od, tomtom_od)

        # Simple linear regression 
        simple_approx_gap = simple_linear_reg(original_od, tomtom_od, zone)
        

        if not explanatory_analyses:
            total_gap['none'][move]  = [(original_gap, 'original')]
            total_gap['none'][move].append((simple_approx_gap, 'simple reg.'))


        for network_property in network_properties:
            if explanatory_analyses:
                total_gap[network_property][move] = [(original_gap, 'original')]
                total_gap[network_property][move].append((simple_approx_gap, 'Reg. simple'))
            for method in methods:
                if explanatory_analyses:
                    # do linear regression with explanatory variables
                    explanatory_approx_gap = explanatory_linear_regression(original_od, tomtom_od, zone, network_property ,method)
                    total_gap[network_property][move].append((explanatory_approx_gap, 'Reg. {} {}'.format(network_property.split('_')[0], method)))
                
                # correlation analysis
                if run_correlation:
                    correlation_analyses(original_od, tomtom_od, network_property, zone, method)

    # Perform the gap function
    if gaps_analysis:
        if not explanatory_analyses:
            gap_bars(total_gap, moves, ['none'])
        else:
            gap_bars(total_gap, moves, network_properties)







main()